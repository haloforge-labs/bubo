// Windows types must precede delayimp; retain this SDK include order.
// clang-format off
#include <windows.h>
// clang-format on
#include <delayimp.h>
#include <xrt/experimental/xrt_version.h>

#include <array>
#include <cstring>
#include <iomanip>
#include <mutex>
#include <span>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "src/core/crypto/sha256.hpp"
#include "src/npu/xrt_runtime.hpp"

namespace gufo::npu {
namespace {

// Retain the verified module for process lifetime. No device/BO is created
// here.
HMODULE runtime_module = nullptr;
std::once_flag runtime_once;

void Require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

class WindowsRuntime final : public detail::RuntimeOperations {
public:
  ~WindowsRuntime() override {
    if (file_ != INVALID_HANDLE_VALUE) {
      CloseHandle(file_);
    }
  }

  RuntimeIdentity Inspect() override {
    std::array<wchar_t, 32768> system{};
    const auto length = GetSystemDirectoryW(system.data(), system.size());
    Require(length && length < system.size(),
            "NPU XRT System32 path unavailable");
    const auto path =
        std::filesystem::path(system.data()) / L"xrt_coreutil.dll";
    // Exclude replacement/writes between identity verification and loading.
    file_ = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                        OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
    Require(file_ != INVALID_HANDLE_VALUE, "NPU XRT DLL unavailable");
    LARGE_INTEGER size{};
    Require(GetFileSizeEx(file_, &size) && size.QuadPart == 2010984,
            "NPU XRT DLL size mismatch");
    std::vector<unsigned char> bytes(static_cast<size_t>(size.QuadPart));
    DWORD count = 0;
    Require(ReadFile(file_, bytes.data(), static_cast<DWORD>(bytes.size()),
                     &count, nullptr) &&
                count == bytes.size(),
            "NPU XRT DLL read failed");
    const std::string digest = crypto::Sha256Hex(
        std::span<const std::uint8_t>(bytes.data(), bytes.size()));
    DWORD ignored = 0;
    const auto version_size = GetFileVersionInfoSizeW(path.c_str(), &ignored);
    std::vector<unsigned char> version(version_size);
    VS_FIXEDFILEINFO* info = nullptr;
    UINT info_size = 0;
    Require(version_size &&
                GetFileVersionInfoW(path.c_str(), 0, version_size,
                                    version.data()) &&
                VerQueryValueW(version.data(), L"\\",
                               reinterpret_cast<void**>(&info), &info_size) &&
                info_size >= sizeof(*info) && info->dwSignature == 0xfeef04bd,
            "NPU XRT DLL file version unavailable");
    return {path,
            digest,
            {HIWORD(info->dwFileVersionMS), LOWORD(info->dwFileVersionMS),
             HIWORD(info->dwFileVersionLS), LOWORD(info->dwFileVersionLS)}};
  }

  void Load(const std::filesystem::path& path) override {
    // Refuse a module preloaded through a foreign search path. Our module was
    // absent before this one explicit load, and its file stays locked above.
    Require(!GetModuleHandleW(L"xrt_coreutil.dll"),
            "NPU XRT DLL was already loaded outside the verified boundary");
    runtime_module =
        LoadLibraryExW(path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    Require(runtime_module != nullptr, "NPU XRT absolute DLL load failed");
  }

  std::array<unsigned, 5> Versions() override {
    return {xrtVersionMajor(), xrtVersionMinor(), xrtVersionPatch(),
            xrtVersionBuild(), xrtVersionCode()};
  }

private:
  HANDLE file_ = INVALID_HANDLE_VALUE;
};

FARPROC WINAPI VerifiedDelayHook(unsigned notification, PDelayLoadInfo info) {
  if (std::strcmp(info->szDll, "xrt_coreutil.dll") != 0) {
    return nullptr;
  }
  if (notification == dliNotePreLoadLibrary) {
    // Never fall back to the delay helper's DLL search. Call Ensure before
    // any XRT API, including construction. A forgotten guard fails closed.
    Require(runtime_module != nullptr,
            "NPU XRT API reached before verified runtime loading");
    return reinterpret_cast<FARPROC>(runtime_module);
  }
  return nullptr;
}

FARPROC WINAPI VerifiedDelayFailure(unsigned, PDelayLoadInfo info) {
  if (std::strcmp(info->szDll, "xrt_coreutil.dll") == 0)
    throw std::runtime_error("NPU XRT delayed import unavailable");
  return nullptr;
}
}  // namespace

void EnsureXrtRuntime() {
  std::call_once(runtime_once, [] {
    WindowsRuntime operations;
    try {
      detail::ValidateAndLoad(operations);
    } catch (...) {
      // A failed version call may have populated delay thunks; unwind those
      // before unloading. No device could have been constructed at this point.
      if (runtime_module) {
        if (!__FUnloadDelayLoadedDLL2("xrt_coreutil.dll")) {
          FreeLibrary(runtime_module);
        }
        runtime_module = nullptr;
      }
      throw;
    }
  });
}

}  // namespace gufo::npu

// Object inclusion (rather than archive extraction) makes the hook mandatory.
extern "C" const PfnDliHook __pfnDliNotifyHook2 = gufo::npu::VerifiedDelayHook;

extern "C" const PfnDliHook __pfnDliFailureHook2 = gufo::npu::VerifiedDelayFailure;
