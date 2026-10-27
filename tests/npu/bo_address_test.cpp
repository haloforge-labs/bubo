// SPDX-License-Identifier: MIT
// CPU only: the BO device-address table, Resolve(), the compact log line and
// the snapshot registry. No XRT, no NPU.
#include <cassert>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>

#include "src/npu/bo_address_table.hpp"

using namespace gufo::npu;

namespace {
bool Has(const std::string& text, const std::string& part) {
  return text.find(part) != std::string::npos;
}
BoEntry E(const char* kind, unsigned index, std::uint64_t address,
          std::uint64_t size, bool known = true) {
  BoEntry e;
  e.kind = kind;
  e.index = index;
  e.address = address;
  e.size = size;
  e.known = known;
  return e;
}
}  // namespace

int main() {
  BoTable table;
  table.entries = {E("weight", 0, 0x1000, 0x1000),
                   E("weight", 1, 0x2000, 0x1000), E("a", 0, 0x10000, 0x100),
                   E("c", 0, 0x20000, 0x100, false)};
  const auto text = [&](std::uint64_t a) { return ResolveText(table, a); };
  // Inside, at both ends, and exactly at the end (= the next BO).
  assert(text(0x1000) == "resolves=weight#0+0x0");
  assert(text(0x1FFF) == "resolves=weight#0+0xFFF");
  assert(text(0x2000) == "resolves=weight#1+0x0");  // boundary: next BO
  assert(text(0x2ABC) == "resolves=weight#1+0xABC");
  assert(text(0x100FF) == "resolves=a#0+0xFF");
  // Outside: the nearest known BO and the distance. The unknown 'c' never
  // counts.
  assert(text(0x3000) == "resolves=none nearest=weight#1 dist=0x1");
  assert(text(0x5000) == "resolves=none nearest=weight#1 dist=0x2001");
  assert(text(0xF00) == "resolves=none nearest=weight#0 dist=0x100");
  assert(text(0x20010) == "resolves=none nearest=a#0 dist=0xFF11");
  // Only the page-rounded value is inside.
  BoTable small;
  small.entries = {E("output", 2, 0x40000, 0x100)};
  assert(ResolveText(small, 0x40800) == "resolves=output#2+0x0 page_rounded=1");
  assert(ResolveText(small, 0x40050) == "resolves=output#2+0x50");
  // Empty table, only unknown addresses.
  assert(ResolveText(BoTable{}, 0x2DEF0000) == "resolves=na");
  BoTable na;
  na.entries = {E("weight", 0, 0, 0x1000, false)};
  assert(ResolveText(na, 0x2DEF0000) == "resolves=na");
  // The faulting values from the 10-13 and 10-15 faults, decimal and hex.
  assert(ParseFaultAddress("770637824") == 0x2DEF0000ULL);
  assert(ParseFaultAddress("773218304") == 0x2E166000ULL);
  assert(ParseFaultAddress("0x2E05B000") == 0x2E05B000ULL);
  assert(!ParseFaultAddress("") && !ParseFaultAddress("xyz") &&
         !ParseFaultAddress("12abc") && !ParseFaultAddress("0x"));
  // Whitespace is trimmed; a sign is rejected (no wrap-around of "-1").
  assert(ParseFaultAddress(" 0x2DEF0000 ") == 0x2DEF0000ULL);
  assert(ParseFaultAddress("\t770637824\n") == 0x2DEF0000ULL);
  assert(ParseFaultAddress("\r\n0x2e05b000\r\n") == 0x2E05B000ULL);
  assert(!ParseFaultAddress("-1") && !ParseFaultAddress("+5") &&
         !ParseFaultAddress("- 5") && !ParseFaultAddress(" -1 ") &&
         !ParseFaultAddress("   ") && !ParseFaultAddress("\t\n") &&
         !ParseFaultAddress("0x -5") && !ParseFaultAddress("1 2"));
  BoTable weights;
  for (unsigned i = 0; i < 100; ++i)
    weights.entries.push_back(
        E("weight", i, 0x2D000000ULL + i * 0x2D0000ULL, 0x2D0000));
  assert(ResolveText(weights, 0x2DEF0000) == "resolves=weight#5+0x" + [] {
    char t[16];
    std::snprintf(t, sizeof(t), "%llX",
                  0x2DEF0000ULL - (0x2D000000ULL + 5 * 0x2D0000ULL));
    return std::string(t);
  }());

  // Compact line: runs, counts, min/max, na.
  BoTable line;
  line.entries = {E("instr", 0, 0x2D000000, 0x6000),
                  E("a", 0, 0x2D006000, 0xB40000),
                  E("scratch", 0, 0x2DB46000, 0x100000),
                  E("scratch", 1, 0x2DC46000, 0x100000),
                  E("weight", 0, 0x30000000, 0x2D0000),
                  E("weight", 1, 0x302D0000, 0x2D0000),
                  E("weight", 2, 0x30700000, 0x2D0000),
                  E("weight", 3, 0, 0x2D0000, false)};
  const auto formatted = FormatBoTable(line);
  std::cout << "BO-LINE: " << formatted << "\n";
  assert(Has(formatted,
             "bo addresses: instr n=1 min=0x2D000000 max=0x2D006000 "
             "runs=[instr#0 0x2D000000+0x6000]"));
  assert(Has(formatted,
             " scratch n=2 min=0x2DB46000 max=0x2DD46000 runs=[scratch#0-1 "
             "0x2DB46000+0x200000]"));
  assert(Has(formatted,
             " weight n=4 na=1 min=0x30000000 max=0x309D0000 runs=[weight#0-1 "
             "0x30000000+0x5A0000,weight#2 0x30700000+0x2D0000]"));
  // Many contiguous weights stay one run; scattered ones are capped at 2 KB.
  BoTable many;
  for (unsigned i = 0; i < 312; ++i)
    many.entries.push_back(
        E("weight", i, 0x30000000ULL + i * 0x2D0000ULL, 0x2D0000));
  const auto one = FormatBoTable(many);
  assert(Has(one, "weight n=312 min=0x30000000") &&
         Has(one, "runs=[weight#0-311 0x30000000+0x"));
  BoTable scattered;
  // 400 scattered weights (~18 KB of runs) exceed the 8 KiB cap.
  for (unsigned i = 0; i < 400; ++i)
    scattered.entries.push_back(
        E("weight", i, 0x30000000ULL + i * 0x400000ULL, 0x2D0000));
  const auto capped = FormatBoTable(scattered);
  assert(kBoLineCapBytes == 8192 && capped.size() <= 8192 &&
         capped.size() > 7000 && Has(capped, "...(+") && Has(capped, " runs)"));
  // 100 scattered weights (~4 KB) now fit untruncated.
  BoTable fits;
  for (unsigned i = 0; i < 100; ++i)
    fits.entries.push_back(
        E("weight", i, 0x30000000ULL + i * 0x400000ULL, 0x2D0000));
  const auto whole = FormatBoTable(fits);
  assert(!Has(whole, "...(+") && Has(whole, "weight#99 "));

  // Snapshot registry: immutable table, generation counts publishes.
  ResetBoRegistryForTest();
  assert(!CurrentBoTable() && BoTableGeneration() == 0);
  auto first = std::make_shared<const BoTable>(table);
  PublishBoTable(first);
  assert(CurrentBoTable() == first && BoTableGeneration() == 1);
  PublishBoTable(std::make_shared<const BoTable>(BoTable{}));
  assert(BoTableGeneration() == 2 &&
         first->entries.size() == 4);  // old stays valid
  // A reader on another thread while the writer republishes (no crash).
  std::thread writer([&] {
    for (int i = 0; i < 2000; ++i)
      PublishBoTable(std::make_shared<const BoTable>(table));
  });
  for (int i = 0; i < 2000; ++i)
    if (const auto t = CurrentBoTable())
      (void)ResolveText(*t, 0x1800);
  writer.join();
  ResetBoRegistryForTest();

  // The latest HAL resolution for the failure line: fresh only.
  ResetHalResolutionForTest();
  assert(RecentHalResolution(3'000'000'000LL).empty());
  NoteHalResolution("resolves=weight#1+0x0");
  assert(RecentHalResolution(3'000'000'000LL) == "resolves=weight#1+0x0");
  std::this_thread::sleep_for(std::chrono::milliseconds(30));
  assert(RecentHalResolution(10'000'000LL).empty());  // older than the window
  ResetHalResolutionForTest();
  std::cout << "PASS BO address table: resolve, parse, line format, snapshot, "
               "recent resolution\n";
}
