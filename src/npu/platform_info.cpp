// SPDX-License-Identifier: MIT
#include "platform_info.hpp"

#include <mutex>

#ifdef _WIN32
#include <windows.h>
#endif

namespace gufo::npu {
namespace {
std::mutex mutex;
PlatformInfo cached;
}  // namespace

std::string SanitizePlatformValue(std::string_view value) {
  std::string out;
  for (const char c : value) {
    const auto u = static_cast<unsigned char>(c);
    if (u < 0x20 || u == 0x7f)
      continue;
    // Non-ASCII bytes become '?': the 64-character cap can never split a UTF-8
    // sequence.
    out += u >= 0x80 ? '?' : (c == ' ' ? '_' : c);
    if (out.size() >= 64)
      break;
  }
  return out.empty() ? "na" : out;
}

std::string FormatPlatformFields(const PlatformInfo& info) {
  return "bios_vendor=" + SanitizePlatformValue(info.bios_vendor) +
         " bios_version=" + SanitizePlatformValue(info.bios_version) +
         " bios_date=" + SanitizePlatformValue(info.bios_date) +
         " board=" + SanitizePlatformValue(info.board) +
         " board_manufacturer=" +
         SanitizePlatformValue(info.board_manufacturer);
}

const PlatformInfo& LoadPlatformInfo(IPlatformSource* source,
                                     std::string* problem) {
  std::lock_guard lock(mutex);
  if (cached.read)
    return cached;
  PlatformInfo info;
  info.read = true;
  try {
    std::string error;
    if (!source || !source->Read(&info, &error))
      *problem = error.empty() ? "no platform source" : error;
    else if (info.bios_vendor == "na" || info.bios_version == "na" ||
             info.bios_date == "na" || info.board == "na" ||
             info.board_manufacturer == "na")
      *problem = "some BIOS/board values are missing";
  } catch (...) {
    *problem = "exception while reading the platform";
  }
  info.read = true;  // the source may have overwritten the whole struct
  cached = info;
  return cached;
}
const PlatformInfo& CachedPlatformInfo() noexcept {
  return cached;  // written once at start, before any chunk
}
void SetPlatformInfoForTest(const PlatformInfo& info) {
  std::lock_guard lock(mutex);
  cached = info;
  cached.read = true;
}
void ResetPlatformInfoForTest() {
  std::lock_guard lock(mutex);
  cached = PlatformInfo{};
}

#ifdef _WIN32
namespace {
class RegistryPlatformSource final : public IPlatformSource {
public:
  bool Read(PlatformInfo* info, std::string* error) override {
    int found = 0;
    const auto get = [&](const wchar_t* name, std::string* out) {
      wchar_t buffer[256];
      DWORD bytes = sizeof(buffer);
      const LONG status = RegGetValueW(
          HKEY_LOCAL_MACHINE, L"HARDWARE\\DESCRIPTION\\System\\BIOS", name,
          RRF_RT_REG_SZ, nullptr, buffer, &bytes);
      if (status != ERROR_SUCCESS)
        return;
      const int size = WideCharToMultiByte(CP_UTF8, 0, buffer, -1, nullptr, 0,
                                           nullptr, nullptr);
      if (size <= 1)
        return;
      std::string text(static_cast<std::size_t>(size), '\0');
      WideCharToMultiByte(CP_UTF8, 0, buffer, -1, text.data(), size, nullptr,
                          nullptr);
      text.resize(static_cast<std::size_t>(size) - 1);
      *out = text;
      ++found;
    };
    get(L"BIOSVendor", &info->bios_vendor);
    get(L"BIOSVersion", &info->bios_version);
    get(L"BIOSReleaseDate", &info->bios_date);
    get(L"BaseBoardProduct", &info->board);
    get(L"BaseBoardManufacturer", &info->board_manufacturer);
    if (!found)
      *error =
          "registry key HKLM\\HARDWARE\\DESCRIPTION\\System\\BIOS not readable";
    return found > 0;
  }
};
}  // namespace
std::unique_ptr<IPlatformSource> MakeRegistryPlatformSource() {
  return std::make_unique<RegistryPlatformSource>();
}
#else
std::unique_ptr<IPlatformSource> MakeRegistryPlatformSource() {
  return nullptr;
}
#endif
}  // namespace gufo::npu
