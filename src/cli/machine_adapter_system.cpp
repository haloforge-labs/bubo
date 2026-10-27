// Real machine probe. Read-only: it calls no Set*, Write* or registration
// API that changes power state, Variable Graphics Memory, firmware or BIOS.

#include <limits>
#include <memory>
#include <string>
#include <utility>

#include "src/cli/machine_adapter.hpp"
#include "src/cli/machine_adapter_adlx.hpp"
#include "src/cli/machine_output.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <dxgi1_4.h>
#include <windows.h>

#include <cstdio>
#include <vector>
#endif

namespace bubo::cli {
namespace {
using gufo::json::Value;

#ifdef _WIN32
std::string Narrow(const std::wstring& text) {
  if (text.empty())
    return {};
  const int size = WideCharToMultiByte(CP_UTF8, 0, text.data(),
                                       static_cast<int>(text.size()), nullptr,
                                       0, nullptr, nullptr);
  std::string out(static_cast<std::size_t>(size), '\0');
  WideCharToMultiByte(CP_UTF8, 0, text.data(), static_cast<int>(text.size()),
                      out.data(), size, nullptr, nullptr);
  return out;
}

std::string GuidText(const GUID& g) {
  char buffer[64];
  std::snprintf(buffer, sizeof(buffer),
                "%08lx-%04x-%04x-%02x%02x-%02x%02x%02x%02x%02x%02x",
                static_cast<unsigned long>(g.Data1), g.Data2, g.Data3,
                g.Data4[0], g.Data4[1], g.Data4[2], g.Data4[3], g.Data4[4],
                g.Data4[5], g.Data4[6], g.Data4[7]);
  return buffer;
}

HMODULE LoadSystemLibrary(const wchar_t* name) {
  return LoadLibraryExW(name, nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
}

std::string Win32Reason(const char* what, long code) {
  return std::string(what) + " returned " + std::to_string(code) + ".";
}

// EFFECTIVE_POWER_MODE notifications deliver the current value to a new
// registration. The callback runs on a system thread.
struct ModeContext {
  HANDLE event = nullptr;
  volatile LONG mode = -1;
};

VOID CALLBACK EffectiveModeCallback(int mode, VOID* context) {
  auto* state = static_cast<ModeContext*>(context);
  InterlockedExchange(&state->mode, mode);
  SetEvent(state->event);
}

// Read-only DXGI enumeration: factory, adapters, GetDesc1 and, when the
// adapter supports it, QueryVideoMemoryInfo. It creates no device.
std::vector<MachineDxgiAdapterInfo> EnumerateDxgi(std::string* error) {
  std::vector<MachineDxgiAdapterInfo> out;
  IDXGIFactory1* factory = nullptr;
  const HRESULT hr = CreateDXGIFactory1(__uuidof(IDXGIFactory1),
                                        reinterpret_cast<void**>(&factory));
  if (FAILED(hr) || !factory) {
    *error = Win32Reason("CreateDXGIFactory1", hr);
    return out;
  }
  for (UINT i = 0;; ++i) {
    IDXGIAdapter1* adapter = nullptr;
    const HRESULT next = factory->EnumAdapters1(i, &adapter);
    if (next == DXGI_ERROR_NOT_FOUND)
      break;
    if (FAILED(next) || !adapter) {
      *error = Win32Reason("EnumAdapters1", next);
      break;
    }
    DXGI_ADAPTER_DESC1 desc{};
    if (SUCCEEDED(adapter->GetDesc1(&desc))) {
      MachineDxgiAdapterInfo info;
      info.name = Narrow(desc.Description);
      info.vendor_id = desc.VendorId;
      info.device_id = desc.DeviceId;
      info.software = (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0;
      info.dedicated_video_memory = desc.DedicatedVideoMemory;
      info.dedicated_system_memory = desc.DedicatedSystemMemory;
      info.shared_system_memory = desc.SharedSystemMemory;
      IDXGIAdapter3* adapter3 = nullptr;
      if (SUCCEEDED(adapter->QueryInterface(
              __uuidof(IDXGIAdapter3), reinterpret_cast<void**>(&adapter3))) &&
          adapter3) {
        info.has_adapter3 = true;
        DXGI_QUERY_VIDEO_MEMORY_INFO memory{};
        const HRESULT q = adapter3->QueryVideoMemoryInfo(
            0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &memory);
        if (SUCCEEDED(q)) {
          info.query_ok = true;
          info.budget = memory.Budget;
          info.current_usage = memory.CurrentUsage;
          info.available_for_reservation = memory.AvailableForReservation;
        } else {
          info.query_error = Win32Reason("QueryVideoMemoryInfo", q);
        }
        DXGI_QUERY_VIDEO_MEMORY_INFO nonlocal{};
        if (SUCCEEDED(adapter3->QueryVideoMemoryInfo(
                0, DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, &nonlocal)))
          info.nonlocal_budget = nonlocal.Budget;
        adapter3->Release();
      }
      out.push_back(std::move(info));
    }
    adapter->Release();
  }
  factory->Release();
  return out;
}

class SystemMachineProbe final : public MachineProbe {
public:
  MachineReading InstalledPhysicalMemory() override {
    ULONGLONG kib = 0;
    const auto time = UtcTime();
    constexpr const char* source = "GetPhysicallyInstalledSystemMemory";
    if (!GetPhysicallyInstalledSystemMemory(&kib) || kib == 0 ||
        kib > std::numeric_limits<std::uint64_t>::max() / 1024)
      return UnavailableReading(source, "Windows", "bytes", time,
                                "Installed physical memory is unavailable.");
    return AvailableReading(static_cast<double>(kib * 1024), source, "Windows",
                            "bytes", time);
  }
  MachineDxgiReadings Dxgi() override {
    std::string error;
    const auto adapters = EnumerateDxgi(&error);
    return DxgiReadingsFromAdapters(adapters, error, UtcTime());
  }

  MachineReading OverlayScheme() override {
    constexpr const char* kSource =
        "powrprof.dll PowerGetEffectiveOverlayScheme";
    const auto time = UtcTime();
    HMODULE lib = LoadSystemLibrary(L"powrprof.dll");
    using Fn = DWORD(WINAPI*)(GUID*);
    auto fn = lib ? reinterpret_cast<Fn>(
                        GetProcAddress(lib, "PowerGetEffectiveOverlayScheme"))
                  : nullptr;
    if (!fn)
      return UnavailableReading(kSource, "Windows 10 1709 or later", "guid",
                                time, "The Windows API is not present.");
    GUID guid{};
    const DWORD rc = fn(&guid);
    if (rc != ERROR_SUCCESS)
      return UnavailableReading(kSource, "Windows 10 1709 or later", "guid",
                                time, Win32Reason("The call", rc));
    return AvailableReading(GuidText(guid), kSource, "Windows 10 1709 or later",
                            "guid", time);
  }

  MachineReading EffectivePowerMode() override {
    constexpr const char* kSource =
        "powrprof.dll PowerRegisterForEffectivePowerModeNotifications "
        "(version 2, EFFECTIVE_POWER_MODE_V2)";
    const auto time = UtcTime();
    HMODULE lib = LoadSystemLibrary(L"powrprof.dll");
    using Register =
        HRESULT(WINAPI*)(ULONG, void(CALLBACK*)(int, VOID*), VOID*, VOID**);
    using Unregister = HRESULT(WINAPI*)(VOID*);
    auto reg =
        lib ? reinterpret_cast<Register>(GetProcAddress(
                  lib, "PowerRegisterForEffectivePowerModeNotifications"))
            : nullptr;
    auto unreg =
        lib ? reinterpret_cast<Unregister>(GetProcAddress(
                  lib, "PowerUnregisterFromEffectivePowerModeNotifications"))
            : nullptr;
    if (!reg || !unreg)
      return UnavailableReading(kSource, "EFFECTIVE_POWER_MODE_V2", "enum",
                                time, "The Windows API is not present.");
    ModeContext context;
    context.event = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (!context.event)
      return UnavailableReading(kSource, "EFFECTIVE_POWER_MODE_V2", "enum",
                                time, "The wait event could not be created.");
    VOID* handle = nullptr;
    const HRESULT hr = reg(2, &EffectiveModeCallback, &context, &handle);
    if (FAILED(hr) || !handle) {
      CloseHandle(context.event);
      return UnavailableReading(kSource, "EFFECTIVE_POWER_MODE_V2", "enum",
                                time, Win32Reason("The registration", hr));
    }
    // Windows sends the current value right after registration.
    const DWORD wait = WaitForSingleObject(context.event, 1000);
    unreg(handle);
    const LONG mode = context.mode;
    CloseHandle(context.event);
    if (wait != WAIT_OBJECT_0 || mode < 0)
      return UnavailableReading(kSource, "EFFECTIVE_POWER_MODE_V2", "enum",
                                time,
                                "Windows did not report a value in 1 second.");
    return AvailableReading(static_cast<int>(mode), kSource,
                            "EFFECTIVE_POWER_MODE_V2", "enum", time);
  }

  MachineReading PowerSource() override {
    constexpr const char* kSource = "kernel32.dll GetSystemPowerStatus";
    const auto time = UtcTime();
    SYSTEM_POWER_STATUS status{};
    if (!GetSystemPowerStatus(&status))
      return UnavailableReading(kSource, "Win32", "text", time,
                                Win32Reason("The call", GetLastError()));
    if (status.ACLineStatus == 1)
      return AvailableReading("ac", kSource, "Win32", "text", time);
    if (status.ACLineStatus == 0)
      return AvailableReading("battery", kSource, "Win32", "text", time);
    return UnavailableReading(kSource, "Win32", "text", time,
                              "Windows reports the power source as unknown.");
  }

  MachineGpuReadings Gpu() override { return Adlx().gpu; }
  MachineVgmReadings Vgm() override { return Adlx().vgm; }

  MachineReading HostPhysicalMemory() override {
    constexpr const char* kSource = "kernel32.dll GlobalMemoryStatusEx";
    const auto time = UtcTime();
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (!GlobalMemoryStatusEx(&status))
      return UnavailableReading(kSource, "Win32", "bytes", time,
                                Win32Reason("The call", GetLastError()));
    return AvailableReading(static_cast<double>(status.ullTotalPhys), kSource,
                            "Win32", "bytes", time);
  }

  MachineReading HostAvailableMemory() override {
    constexpr const char* kSource = "kernel32.dll GlobalMemoryStatusEx";
    const auto time = UtcTime();
    MEMORYSTATUSEX status{};
    status.dwLength = sizeof(status);
    if (!GlobalMemoryStatusEx(&status))
      return UnavailableReading(kSource, "Win32", "bytes", time,
                                Win32Reason("The call", GetLastError()));
    return AvailableReading(static_cast<double>(status.ullAvailPhys), kSource,
                            "Win32", "bytes", time);
  }

  MachineReading OemManufacturer() override {
    return Bios(L"SystemManufacturer");
  }
  MachineReading OemModel() override { return Bios(L"SystemProductName"); }
  MachineReading BiosVendor() override { return Bios(L"BIOSVendor"); }
  MachineReading BiosVersion() override { return Bios(L"BIOSVersion"); }
  MachineReading BiosDate() override { return Bios(L"BIOSReleaseDate"); }

private:
  MachineReading Bios(const wchar_t* name) {
    const std::string source =
        "registry HKLM\\HARDWARE\\DESCRIPTION\\System\\BIOS\\" + Narrow(name);
    const auto time = UtcTime();
    wchar_t buffer[512];
    DWORD size = sizeof(buffer);
    const LONG rc =
        RegGetValueW(HKEY_LOCAL_MACHINE, L"HARDWARE\\DESCRIPTION\\System\\BIOS",
                     name, RRF_RT_REG_SZ, nullptr, buffer, &size);
    if (rc != ERROR_SUCCESS)
      return UnavailableReading(source, "SMBIOS table copy", "text", time,
                                Win32Reason("The registry read", rc));
    std::string text = Narrow(buffer);
    if (text.empty())
      return UnavailableReading(source, "SMBIOS table copy", "text", time,
                                "The registry value is empty.");
    return AvailableReading(std::move(text), source, "SMBIOS table copy",
                            "text", time);
  }

  // One ADLX sample per probe, taken on first use.
  const AdlxSample& Adlx() {
    if (!adlx_)
      adlx_ = std::make_unique<AdlxSample>(SampleAdlx());
    return *adlx_;
  }
  std::unique_ptr<AdlxSample> adlx_;
};
#else
class SystemMachineProbe final : public MachineProbe {
public:
  MachineReading OverlayScheme() override { return None("guid"); }
  MachineReading EffectivePowerMode() override { return None("enum"); }
  MachineReading PowerSource() override { return None("text"); }
  MachineGpuReadings Gpu() override { return Adlx().gpu; }
  MachineVgmReadings Vgm() override { return Adlx().vgm; }
  MachineReading HostPhysicalMemory() override { return None("bytes"); }
  MachineReading OemManufacturer() override { return None("text"); }
  MachineReading OemModel() override { return None("text"); }
  MachineReading BiosVendor() override { return None("text"); }
  MachineReading BiosVersion() override { return None("text"); }
  MachineReading BiosDate() override { return None("text"); }

private:
  static MachineReading None(const char* unit) {
    return UnavailableReading("none", "none", unit, UtcTime(),
                              "This machine adapter reads Windows only.");
  }
  const AdlxSample& Adlx() {
    if (!adlx_)
      adlx_ = std::make_unique<AdlxSample>(SampleAdlx());
    return *adlx_;
  }
  std::unique_ptr<AdlxSample> adlx_;
};
#endif
}  // namespace

std::unique_ptr<MachineProbe> CreateSystemMachineProbe() {
  return std::make_unique<SystemMachineProbe>();
}
}  // namespace bubo::cli
