// SPDX-License-Identifier: MIT
#ifndef GUFO_NPU_PLATFORM_INFO_HPP_
#define GUFO_NPU_PLATFORM_INFO_HPP_
// BIOS and board identification for the NPU logs. Read ONCE at NPU start from
// the SMBIOS copy in the registry (HKLM\HARDWARE\DESCRIPTION\System\BIOS, no
// admin, no COM/WMI) and repeated on the NPU failure line. Fail-open: a missing
// value prints "na", one line is logged, nothing throws.
#include <memory>
#include <string>
#include <string_view>

namespace gufo::npu {
struct PlatformInfo {
  bool read = false;  // a read was attempted
  std::string bios_vendor = "na", bios_version = "na", bios_date = "na",
              board = "na", board_manufacturer = "na";
};
class IPlatformSource {
public:
  virtual ~IPlatformSource() = default;
  // Fills the fields it can; false (with *error) when nothing could be read.
  virtual bool Read(PlatformInfo* info, std::string* error) = 0;
};
// Strip control characters, replace spaces with '_', cap at 64 characters;
// empty -> "na".
std::string SanitizePlatformValue(std::string_view value);
// "bios_vendor=.. bios_version=.. bios_date=.. board=.. board_manufacturer=..".
std::string FormatPlatformFields(const PlatformInfo& info);
// Windows registry source (read only). Null on other platforms.
std::unique_ptr<IPlatformSource> MakeRegistryPlatformSource();
// Reads through the source once and caches the result (later calls keep it).
// *problem receives a short text when something was missing (log it once).
const PlatformInfo& LoadPlatformInfo(IPlatformSource* source,
                                     std::string* problem);
// The cached info (all "na" before a load).
const PlatformInfo& CachedPlatformInfo() noexcept;
// Tests: replace or clear the cache.
void SetPlatformInfoForTest(const PlatformInfo& info);
void ResetPlatformInfoForTest();
}  // namespace gufo::npu
#endif
