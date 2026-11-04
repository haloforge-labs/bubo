#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_DECODE_CYCLE_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_DECODE_CYCLE_HPP_

#include <algorithm>
#include <array>
#include <charconv>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace gufo::models::qwen38_flash_next {

// kFixed2..kFixed7 (debug hooks only) always propose that many drafts, up to the cap.
// kFitNoProbe (debug hooks only) is kFit without its deeper-draft probe.
enum class DraftPolicy { kDefault, kFit, kOff, kSerial64, kFitNoProbe, kFixed2, kFixed3, kFixed4, kFixed5, kFixed6, kFixed7 };
enum class DraftFalsify { kNone, kWrong, kShift, kRepeat };

inline bool ParseLogitEvalDump(const char* value, const char* debug,
                               bool* enabled, std::string* error) {
  *enabled = false;
  if (!value || std::string_view(value) == "0")
    return true;
  if ((std::string_view(value) != "1" && std::string_view(value) != "heads") ||
      !debug || std::string_view(debug) != "1") {
    if (error)
      *error =
          "GUFO_LOGIT_EVAL_DUMP must be 0, 1 or heads; enabled dumps require "
          "GUFO_DEBUG_HOOKS=1";
    return false;
  }
  *enabled = true;
  return true;
}

struct DecodeCycleOptions {
  bool ple_readahead{true};
  bool device_accept{false};
  bool draft_chain{true};
  bool moe_fusion{false};
  bool hc_inject_bf16_fused{true};
  DraftPolicy policy{DraftPolicy::kDefault};
  DraftFalsify falsify{DraftFalsify::kNone};
  std::uint32_t falsify_cycles{32};
};

/// Literal env parsing is shared with the CPU comparison; empty and unknown
/// values are errors. Unset knobs enable read-ahead and the current draft policy
/// (fit is opt-in: it lost 15 % on the production package, 2026-11-01).
inline bool ParseDecodeCycleOptions(const char* ahead, const char* policy,
                                    const char* falsify, const char* debug,
                                    const char* cycles, DecodeCycleOptions* out,
                                    std::string* error,
                                    const char* device_accept = nullptr,
                                    const char* draft_chain = nullptr,
                                    const char* moe_fusion = nullptr,
                                    const char* hc_inject_bf16_fused = nullptr) {
  const auto fail = [&](const char* message) {
    if (error)
      *error = message;
    return false;
  };
  DecodeCycleOptions parsed;
  const auto boolean = [&](const char* text, bool* result) {
    if (!text)
      return true;
    const std::string_view value(text);
    if (value != "0" && value != "1")
      return false;
    *result = value == "1";
    return true;
  };
  if (!boolean(ahead, &parsed.ple_readahead))
    return fail("GUFO_PLE_READAHEAD must be 0 or 1");
  if (!boolean(device_accept, &parsed.device_accept))
    return fail("GUFO_DEVICE_ACCEPT must be 0 or 1");
  if (!boolean(draft_chain, &parsed.draft_chain))
    return fail("GUFO_DRAFT_CHAIN must be 0 or 1");
  if (!boolean(moe_fusion, &parsed.moe_fusion))
    return fail("GUFO_MOE_FUSION must be 0 or 1");
  if (!boolean(hc_inject_bf16_fused, &parsed.hc_inject_bf16_fused))
    return fail("GUFO_HC_INJECT_BF16_FUSED must be 0 or 1");
  if (policy) {
    const std::string_view value(policy);
    if (value == "current")
      parsed.policy = DraftPolicy::kDefault;
    else if (value == "fit")
      parsed.policy = DraftPolicy::kFit;
    else if (value == "off")
      parsed.policy = DraftPolicy::kOff;
    else if (value == "serial64")
      parsed.policy = DraftPolicy::kSerial64;
    else if (value == "fitnoprobe") {
      if (!debug || std::string_view(debug) != "1")
        return fail("GUFO_DRAFT_POLICY fitnoprobe requires GUFO_DEBUG_HOOKS=1");
      parsed.policy = DraftPolicy::kFitNoProbe;
    } else if (value.size() == 6 && value.substr(0, 5) == "fixed" &&
             value[5] >= '2' && value[5] <= '7') {
      if (!debug || std::string_view(debug) != "1")
        return fail("GUFO_DRAFT_POLICY fixed2..fixed7 requires GUFO_DEBUG_HOOKS=1");
      parsed.policy = static_cast<DraftPolicy>(
          static_cast<int>(DraftPolicy::kFixed2) + (value[5] - '2'));
    } else
      return fail("GUFO_DRAFT_POLICY must be current, fit, off or serial64");
  }
  if (falsify) {
    if (!debug || std::string_view(debug) != "1")
      return fail("GUFO_DRAFT_FALSIFY requires GUFO_DEBUG_HOOKS=1");
    const std::string_view value(falsify);
    if (value == "wrong")
      parsed.falsify = DraftFalsify::kWrong;
    else if (value == "shift")
      parsed.falsify = DraftFalsify::kShift;
    else if (value == "repeat")
      parsed.falsify = DraftFalsify::kRepeat;
    else
      return fail("GUFO_DRAFT_FALSIFY must be wrong, shift or repeat");
  }
  if (cycles) {
    const std::string_view value(cycles);
    const auto [end, ec] = std::from_chars(
        value.data(), value.data() + value.size(), parsed.falsify_cycles);
    if (!falsify || ec != std::errc{} || end != value.data() + value.size() ||
        parsed.falsify_cycles == 0 || parsed.falsify_cycles > 1000000)
      return fail(
          "GUFO_DRAFT_FALSIFY_CYCLES needs a falsifier and an integer "
          "from 1 through 1000000");
  }
  *out = parsed;
  return true;
}

/// C_d = 25.34 + 5.05*(d+1) + 1.83*d ms. Propose the next token only if
/// survival > expected_tokens * marginal_cost / current_cycle_cost.
inline std::uint32_t FitDraftDepth(std::span<const float> conditional,
                                   std::uint32_t budget) noexcept {
  constexpr double base_ms = 25.34 + 5.05;
  constexpr double extra_ms = 1.83 + 5.05;
  double survival = 1, expected = 1;
  std::uint32_t depth = 0;
  const auto cap = std::min<std::size_t>(budget, conditional.size());
  for (; depth < cap; ++depth) {
    survival *= std::clamp(conditional[depth], 0.0F, 1.0F);
    if (!(survival > expected * extra_ms / (base_ms + extra_ms * depth)))
      break;
    expected += survival;
  }
  return depth;
}

/// Each byte holds accepted (low nibble) and offered (high nibble), at most
/// seven each. Serial tokens do not count as acceptance rounds.
struct DraftRoundWindow {
  std::array<std::uint8_t, 32> rounds{};
  std::uint32_t count{0}, next{0}, serial_left{0};
  std::uint32_t probe{0};
  std::uint32_t mode{0};

  bool Observe(std::uint32_t accepted, std::uint32_t offered) noexcept {
    rounds[next] = static_cast<std::uint8_t>((offered << 4) | accepted);
    next = (next + 1) % rounds.size();
    count = std::min(count + 1, 32U);
    std::uint32_t kept = 0, total = 0;
    for (std::uint32_t i = 0; i < count; ++i) {
      kept += rounds[i] & 15U;
      total += rounds[i] >> 4;
    }
    if (count == 32 && 100U * kept < 35U * total) {
      serial_left = 64;
      return true;
    }
    return false;
  }
};

struct DraftPolicyTotals {
  std::uint64_t rounds{0}, serial_tokens{0};
  std::array<std::uint64_t, 8> depth_hist{};
  void Observe(std::uint32_t depth, std::size_t tokens) noexcept {
    ++rounds;
    ++depth_hist[std::min<std::size_t>(depth, depth_hist.size() - 1)];
    if (depth == 0)
      serial_tokens += tokens;
  }
};

/// A private deterministic stream leaves target and draft RNG untouched.
inline std::int32_t FalsifyDraftToken(DraftFalsify mode, std::int32_t original,
                                      std::int32_t anchor, std::uint32_t vocab,
                                      std::uint32_t base,
                                      std::uint32_t depth) noexcept {
  if (vocab < 2 || mode == DraftFalsify::kNone)
    return original;
  if (mode == DraftFalsify::kRepeat)
    return anchor;
  std::uint64_t shift = 1;
  if (mode == DraftFalsify::kWrong) {
    std::uint64_t mixed = (std::uint64_t{base} << 32) | depth;
    mixed += 0x9e3779b97f4a7c15ULL;
    mixed = (mixed ^ (mixed >> 30)) * 0xbf58476d1ce4e5b9ULL;
    mixed = (mixed ^ (mixed >> 27)) * 0x94d049bb133111ebULL;
    shift += (mixed ^ (mixed >> 31)) % (vocab - 1);
  }
  return static_cast<std::int32_t>(
      (std::uint64_t{static_cast<std::uint32_t>(original)} + shift) % vocab);
}

}  // namespace gufo::models::qwen38_flash_next
#endif  // GUFO_MODELS_QWEN38_FLASH_NEXT_DECODE_CYCLE_HPP_
