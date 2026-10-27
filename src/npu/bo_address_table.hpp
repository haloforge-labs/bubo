// SPDX-License-Identifier: MIT
#ifndef GUFO_NPU_BO_ADDRESS_TABLE_HPP_
#define GUFO_NPU_BO_ADDRESS_TABLE_HPP_
// Device addresses of every XRT buffer object the NPU path owns, so that the
// next IOMMU fault can name the buffer it hit (the 10-13 and 10-15 faults both
// reported 0x2DEF0000). Log-only and fail-open: an address that cannot be read
// is recorded as unknown ("na"), nothing here throws, nothing changes NPU work.
// Header only (inline), so every target that compiles the adapter or the HAL
// monitor gets it without new link inputs.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "fault_state.hpp"

namespace gufo::npu {
struct BoEntry {
  const char* kind =
      "";  // instr|a|b|c|scratch|heartbeat_a|heartbeat_b|output|weight
  unsigned index = 0;  // index inside the kind (creation order)
  std::uint64_t size = 0;
  std::uint64_t address = 0;
  bool known = false;  // false: xrt::bo::address() failed or returned 0
};
struct BoTable {
  std::vector<BoEntry> entries;
};

// Result of Resolve(): inside a BO, or the nearest one, or nothing known.
struct BoResolution {
  bool any_known = false;
  bool inside = false;
  bool page_rounded = false;       // only the 4 KiB-rounded address was inside
  const BoEntry* entry = nullptr;  // the BO inside, or the nearest one
  std::uint64_t offset = 0;        // inside: address - entry.address
  std::uint64_t distance = 0;      // nearest: bytes from the closest edge
};
inline BoResolution Resolve(const BoTable& table, std::uint64_t address) {
  BoResolution out;
  const auto inside = [&](std::uint64_t value) -> const BoEntry* {
    for (const auto& e : table.entries)
      if (e.known && value >= e.address && value - e.address < e.size)
        return &e;
    return nullptr;
  };
  for (const auto& e : table.entries)
    out.any_known = out.any_known || e.known;
  if (!out.any_known)
    return out;
  if (const auto* e = inside(address)) {
    out.inside = true;
    out.entry = e;
    out.offset = address - e->address;
    return out;
  }
  if (const auto* e = inside(address & ~std::uint64_t{0xFFF})) {
    out.inside = true;
    out.page_rounded = true;
    out.entry = e;
    out.offset = (address & ~std::uint64_t{0xFFF}) - e->address;
    return out;
  }
  std::uint64_t best = ~std::uint64_t{0};
  for (const auto& e : table.entries) {
    if (!e.known)
      continue;
    const std::uint64_t end = e.address + e.size;
    const std::uint64_t d =
        address < e.address ? e.address - address : address - end + 1;
    if (d < best) {
      best = d;
      out.entry = &e;
      out.distance = d;
    }
  }
  return out;
}
inline std::string Hex(std::uint64_t v) {
  char text[24];
  std::snprintf(text, sizeof(text), "0x%llX",
                static_cast<unsigned long long>(v));
  return text;
}
inline std::string Name(const BoEntry& e) {
  return std::string(e.kind) + "#" + std::to_string(e.index);
}
// "resolves=weight#41+0x1234", "resolves=weight#41+0x0 page_rounded=1",
// "resolves=none nearest=weight#41 dist=0x1000" or "resolves=na" (no known BO).
inline std::string ResolveText(const BoTable& table, std::uint64_t address) {
  const auto r = Resolve(table, address);
  if (!r.any_known)
    return "resolves=na";
  if (r.inside)
    return "resolves=" + Name(*r.entry) + "+" + Hex(r.offset) +
           (r.page_rounded ? " page_rounded=1" : "");
  return "resolves=none nearest=" + Name(*r.entry) + " dist=" + Hex(r.distance);
}
// FaultInformation text from the event record: "0x2E05B000" or decimal digits.
inline std::optional<std::uint64_t> ParseFaultAddress(const std::string& text) {
  // Trim ASCII whitespace; a sign is never valid (stoull would wrap "-1").
  const auto is_space = [](char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' ||
           c == '\f';
  };
  std::size_t b = 0, e = text.size();
  while (b < e && is_space(text[b]))
    ++b;
  while (e > b && is_space(text[e - 1]))
    --e;
  const std::string t = text.substr(b, e - b);
  if (t.empty() || t[0] == '-' || t[0] == '+')
    return std::nullopt;
  try {
    std::size_t used = 0;
    const bool hex =
        t.size() > 2 && t[0] == '0' && (t[1] == 'x' || t[1] == 'X');
    const auto v = std::stoull(t, &used, hex ? 16 : 10);
    if (used != t.size())
      return std::nullopt;
    return v;
  } catch (...) {
    return std::nullopt;
  }
}

// ---- compact log line (8 KiB at most) -------------------------------------
inline constexpr std::size_t kBoLineCapBytes = 8192;
// "bo addresses: <kind> n=<count> [na=<unknown>] min=0x.. max=0x..
// runs=[kind#first-last 0xADDR+0xBYTES,...]" repeated per kind. A run is a
// stretch of consecutive indices whose addresses follow each other without a
// gap.
inline std::string FormatBoTable(const BoTable& table,
                                 std::size_t cap_bytes = kBoLineCapBytes) {
  static constexpr const char* kKinds[] = {
      "instr",       "a",           "b",      "c",     "scratch",
      "heartbeat_a", "heartbeat_b", "output", "weight"};
  std::string out = "bo addresses:";
  for (const char* kind : kKinds) {
    std::vector<const BoEntry*> list;
    for (const auto& e : table.entries)
      if (std::string(e.kind) == kind)
        list.push_back(&e);
    if (list.empty())
      continue;
    std::sort(list.begin(), list.end(), [](const BoEntry* a, const BoEntry* b) {
      return a->index < b->index;
    });
    std::size_t unknown = 0;
    std::uint64_t lo = ~std::uint64_t{0}, hi = 0;
    for (const auto* e : list) {
      if (!e->known) {
        ++unknown;
        continue;
      }
      lo = std::min(lo, e->address);
      hi = std::max(hi, e->address + e->size);
    }
    std::string head =
        std::string(" ") + kind + " n=" + std::to_string(list.size());
    if (unknown)
      head += " na=" + std::to_string(unknown);
    if (unknown < list.size())
      head += " min=" + Hex(lo) + " max=" + Hex(hi) + " runs=[";
    else
      head += " runs=[";
    std::string runs;
    std::size_t run_count = 0, shown = 0;
    for (std::size_t i = 0; i < list.size();) {
      if (!list[i]->known) {
        ++i;
        continue;
      }
      std::size_t j = i;
      std::uint64_t bytes = list[i]->size;
      while (j + 1 < list.size() && list[j + 1]->known &&
             list[j + 1]->index == list[j]->index + 1 &&
             list[j + 1]->address == list[j]->address + list[j]->size) {
        ++j;
        bytes += list[j]->size;
      }
      ++run_count;
      std::string run = kind;
      run += "#" + std::to_string(list[i]->index);
      if (j > i)
        run += "-" + std::to_string(list[j]->index);
      run += " " + Hex(list[i]->address) + "+" + Hex(bytes);
      if (out.size() + head.size() + runs.size() + run.size() + 40 <
          cap_bytes) {
        runs += (runs.empty() ? "" : ",") + run;
        ++shown;
      }
      i = j + 1;
    }
    if (shown < run_count)
      runs += ",...(+" + std::to_string(run_count - shown) + " runs)";
    out += head + runs + "]";
  }
  return out.size() > cap_bytes ? out.substr(0, cap_bytes) : out;
}

// ---- process-wide snapshot (immutable table published by the adapter) -----
struct BoRegistry {
  std::mutex mutex;
  std::shared_ptr<const BoTable> table;
  std::uint64_t generation = 0;
};
inline BoRegistry& Registry() {
  static BoRegistry registry;
  return registry;
}
inline void PublishBoTable(std::shared_ptr<const BoTable> table) noexcept {
  try {
    auto& r = Registry();
    std::lock_guard lock(r.mutex);
    r.table = std::move(table);
    ++r.generation;
  } catch (...) {
  }
}
inline std::shared_ptr<const BoTable> CurrentBoTable() noexcept {
  try {
    auto& r = Registry();
    std::lock_guard lock(r.mutex);
    return r.table;
  } catch (...) {
    return nullptr;
  }
}
inline std::uint64_t BoTableGeneration() noexcept {
  try {
    auto& r = Registry();
    std::lock_guard lock(r.mutex);
    return r.generation;
  } catch (...) {
    return 0;
  }
}
// Test hook.
inline void ResetBoRegistryForTest() {
  auto& r = Registry();
  std::lock_guard lock(r.mutex);
  r.table.reset();
  r.generation = 0;
}

// ---- the latest HAL fault's resolution, for the NPU failure line ----------
struct HalResolutionSlot {
  std::mutex mutex;
  std::string text;
  std::int64_t ts_ns = 0;
};
inline HalResolutionSlot& HalSlot() {
  static HalResolutionSlot slot;
  return slot;
}
inline void NoteHalResolution(const std::string& text) noexcept {
  try {
    auto& s = HalSlot();
    std::lock_guard lock(s.mutex);
    s.text = text;
    s.ts_ns = SteadyNowNs();
  } catch (...) {
  }
}
// The latest resolution if it is younger than window_ns; empty otherwise.
inline std::string RecentHalResolution(std::int64_t window_ns) noexcept {
  try {
    auto& s = HalSlot();
    std::lock_guard lock(s.mutex);
    if (s.ts_ns != 0 && SteadyNowNs() - s.ts_ns <= window_ns)
      return s.text;
  } catch (...) {
  }
  return {};
}
inline void ResetHalResolutionForTest() {
  auto& s = HalSlot();
  std::lock_guard lock(s.mutex);
  s.text.clear();
  s.ts_ns = 0;
}
}  // namespace gufo::npu
#endif
