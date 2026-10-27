// SPDX-License-Identifier: MIT
// CPU only: BIOS/board identification for the NPU logs. Fake source for the
// formatting, sanitising, fail-open and cache-once rules; one read-only look at
// the real registry values (printed, never required to be present).
#include "src/npu/platform_info.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>

#include "src/npu/fault_telemetry.hpp"

using namespace gufo::npu;

namespace {
bool Has(const std::string& text, const std::string& part) {
  return text.find(part) != std::string::npos;
}
struct Fake final : IPlatformSource {
  PlatformInfo values;
  bool ok = true, throws = false;
  std::string error = "registry key not readable";
  int reads = 0;
  bool Read(PlatformInfo* info, std::string* out_error) override {
    ++reads;
    if (throws)
      throw std::runtime_error("boom");
    if (!ok) {
      *out_error = error;
      return false;
    }
    *info = values;
    return true;
  }
};
Fake Asus() {
  Fake f;
  f.values.bios_vendor = "American Megatrends International, LLC.";
  f.values.bios_version = "HN7306EAC.310";
  f.values.bios_date = "07/08/2026";
  f.values.board = "HN7306EAC";
  f.values.board_manufacturer = "ASUSTeK COMPUTER INC.";
  return f;
}
}  // namespace

int main() {
  // Sanitising: control characters out, spaces to '_', cap 64, empty = na.
  assert(SanitizePlatformValue("a b\tc\r\nd\x01\x7f") == "a_bcd");
  assert(SanitizePlatformValue("") == "na" &&
         SanitizePlatformValue("\x01\x02") == "na");
  assert(SanitizePlatformValue(std::string(200, 'x')) == std::string(64, 'x'));
  assert(SanitizePlatformValue("  ") == "__");
  // Non-ASCII bytes never leave half a UTF-8 sequence behind.
  assert(SanitizePlatformValue("caf\xc3\xa9") == "caf??");
  assert(SanitizePlatformValue(std::string(63, 'x') + "\xc3\xa9") ==
         std::string(63, 'x') + "?");

  // Format.
  {
    auto f = Asus();
    const auto line = FormatPlatformFields(f.values);
    std::cout << "PLATFORM-FIELDS: " << line << "\n";
    assert(line ==
           "bios_vendor=American_Megatrends_International,_LLC. "
           "bios_version=HN7306EAC.310 bios_date=07/08/2026 board=HN7306EAC "
           "board_manufacturer=ASUSTeK_COMPUTER_INC.");
  }
  // Not read yet: every field is na (and the failure line still has them).
  ResetPlatformInfoForTest();
  assert(FormatPlatformFields(CachedPlatformInfo()) ==
         "bios_vendor=na bios_version=na bios_date=na board=na "
         "board_manufacturer=na");
  // Load once and cache: a second source is ignored.
  {
    auto f = Asus();
    std::string problem;
    const auto& info = LoadPlatformInfo(&f, &problem);
    assert(problem.empty() && info.read &&
           info.bios_version == "HN7306EAC.310");
    auto other = Asus();
    other.values.bios_version = "other";
    std::string p2;
    assert(LoadPlatformInfo(&other, &p2).bios_version == "HN7306EAC.310" &&
           other.reads == 0);
    // The failure line carries the fields.
    ChunkStart start;
    ChunkEnd end;
    const auto failure = FormatChunkFailure(start, end, kInvalidDecoderExp255);
    assert(Has(failure,
               " bios_vendor=American_Megatrends_International,_LLC. "
               "bios_version=HN7306EAC.310 bios_date=07/08/2026 "
               "board=HN7306EAC board_manufacturer=ASUSTeK_COMPUTER_INC."));
  }
  // Partial read: the missing values are na and the problem is reported.
  {
    ResetPlatformInfoForTest();
    auto f = Asus();
    f.values.bios_date = "na";
    std::string problem;
    const auto& info = LoadPlatformInfo(&f, &problem);
    assert(info.bios_date == "na" && Has(problem, "missing"));
    ResetPlatformInfoForTest();
    auto g = Asus();
    g.values.board_manufacturer = "na";  // the board manufacturer counts too
    std::string problem2;
    LoadPlatformInfo(&g, &problem2);
    assert(Has(problem2, "missing"));
  }
  // Total failure and a throwing source: fail-open, na, never throws.
  {
    ResetPlatformInfoForTest();
    Fake f;
    f.ok = false;
    std::string problem;
    const auto& info = LoadPlatformInfo(&f, &problem);
    assert(info.read && info.bios_version == "na" &&
           Has(problem, "not readable"));
    ResetPlatformInfoForTest();
    Fake t;
    t.throws = true;
    std::string p2;
    assert(LoadPlatformInfo(&t, &p2).bios_version == "na" && !p2.empty());
    ResetPlatformInfoForTest();
    std::string p3;
    assert(LoadPlatformInfo(nullptr, &p3).bios_vendor == "na" && !p3.empty());
  }
  // The real registry values on this machine: read only, printed.
  {
    ResetPlatformInfoForTest();
    auto real = MakeRegistryPlatformSource();
    std::string problem;
    const auto& info = LoadPlatformInfo(real.get(), &problem);
    std::cout << "REAL-REGISTRY: " << FormatPlatformFields(info)
              << (problem.empty() ? "" : " | problem: " + problem) << "\n";
    ResetPlatformInfoForTest();
  }
  std::cout << "PASS platform info: sanitising, format, cache once, fail-open, "
               "failure-line fields\n";
}
