#pragma once

#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::qwen38_flash_next::hc_load_q8 {

// GUFO_HC_LOAD_Q8, process lifetime, default ON (0 keeps the BF16 weights). 1 narrows the BF16
// hyper-connection down/up projections to Q8_0 (and widens a BF16 inject to
// F32) in memory while the weights are uploaded, so a native IQ/K-quant file
// takes the route a production package takes. The GGUF is never changed.
// Like GUFO_HC_BF16_DENSE an unset variable means 1; the debug switches below
// (VERIFY, POISON) keep unset = 0 through Parse.
constexpr int Parse(const char* value) {
  if (!value || (value[0] == '0' && value[1] == '\0'))
    return 0;
  if (value[0] == '1' && value[1] == '\0')
    return 1;
  return -1;
}

/// Same values as Parse, but unset means 1.
constexpr int ParseDefaultOn(const char* value) {
  return value ? Parse(value) : 1;
}

inline bool Enabled() {
  static const bool enabled = [] {
    const int parsed = ParseDefaultOn(std::getenv("GUFO_HC_LOAD_Q8"));
    if (parsed < 0)
      throw std::invalid_argument("unknown GUFO_HC_LOAD_Q8; expected 0 or 1");
    return parsed == 1;
  }();
  return enabled;
}

// GUFO_HC_LOAD_Q8_VERIFY (debug, process lifetime, default 0, same 0/1 rule):
// after the upload read the converted tensors and the whole MTP block back
// from the device, compare them with the file and print one [hcq8-verify] line.
inline bool VerifyEnabled() {
  static const bool enabled = [] {
    const int parsed = Parse(std::getenv("GUFO_HC_LOAD_Q8_VERIFY"));
    if (parsed < 0)
      throw std::invalid_argument(
          "unknown GUFO_HC_LOAD_Q8_VERIFY; expected 0 or 1");
    return parsed == 1;
  }();
  return enabled;
}

// GUFO_HC_LOAD_Q8_POISON (debug, process lifetime, default 0, same 0/1 rule,
// independent of GUFO_HC_LOAD_Q8): every buffer the loader releases or replaces
// is overwritten with kPoisonByte when it is released; a device buffer is then
// not freed (it stays allocated and poisoned), pinned staging memory is kept.
// A stale reader sees NaN instead of the old data, deterministically.
inline bool PoisonEnabled() {
  static const bool enabled = [] {
    const int parsed = Parse(std::getenv("GUFO_HC_LOAD_Q8_POISON"));
    if (parsed < 0)
      throw std::invalid_argument(
          "unknown GUFO_HC_LOAD_Q8_POISON; expected 0 or 1");
    return parsed == 1;
  }();
  return enabled;
}

/// 0xFF in every byte: 0xFFFF is a NaN in BF16 and in F16, 0xFFFFFFFF a NaN in
/// F32, and in a Q8_0 block the binary16 scale is 0xFFFF (NaN) over int8 -1.
constexpr std::uint8_t kPoisonByte = 0xFF;

/// What the poisoning touched, for the one [loader-poison] line.
struct PoisonLedger {
  std::size_t host_events{0}, host_bytes{0};      ///< host buffers overwritten
  std::size_t kept_device{0}, kept_device_bytes{0};  ///< device buffers poisoned and not freed
  std::size_t kept_host_bytes{0};  ///< pinned memory poisoned and not freed
  /// Overwrites a host buffer at the moment it is released.
  void Host(void* data, std::size_t bytes) {
    if (data != nullptr && bytes != 0) {
      std::memset(data, kPoisonByte, bytes);
      ++host_events;
      host_bytes += bytes;
    }
  }
  /// Accounts a device buffer the caller poisoned (hipMemset) and kept.
  void KeptDevice(std::size_t bytes) {
    ++kept_device;
    kept_device_bytes += bytes;
  }
};

enum class Role : std::uint8_t { kDown, kUp, kInject };

/// Where a mixer sits. The trunk groups repeat once per layer.
enum class Group : std::uint8_t {
  kTrunkAttn,
  kTrunkFfn,
  kMtpAttn,
  kMtpFfn,
  kMtpHead,
  kOutputHead,
  kCount
};

constexpr std::size_t kGroups = static_cast<std::size_t>(Group::kCount);

constexpr const char* Name(Group g) {
  switch (g) {
    case Group::kTrunkAttn: return "trunk_attn";
    case Group::kTrunkFfn: return "trunk_ffn";
    case Group::kMtpAttn: return "mtp_attn";
    case Group::kMtpFfn: return "mtp_ffn";
    case Group::kMtpHead: return "mtp_head";
    case Group::kOutputHead: return "output_head";
    default: return "?";
  }
}

constexpr const char* Name(Role r) {
  return r == Role::kDown ? "down" : r == Role::kUp ? "up" : "inject";
}

/// One HC tensor of one mixer, as the binder saw it. `mixer` is unique per
/// mixer; its down, up (and inject, when it has one) share it.
struct Tensor {
  std::uint32_t mixer{0};
  Group group{Group::kTrunkAttn};
  Role role{Role::kDown};
  core::GgmlType type{core::GgmlType::kF32};
  std::uint64_t rows{0};
  std::uint64_t cols{0};
};

/// BF16 -> F32 is exact: the 16 stored bits become the high half of the F32.
inline void WidenBf16(const void* bf16, std::size_t count, float* out) {
  const auto* source = static_cast<const std::uint8_t*>(bf16);
  for (std::size_t i = 0; i < count; ++i) {
    std::uint16_t bits;
    std::memcpy(&bits, source + i * 2, 2);
    out[i] = std::bit_cast<float>(std::uint32_t{bits} << 16);
  }
}

/// Host threads the loader quantises one tensor on (the weight stager keeps
/// its own I/O threads busy meanwhile).
constexpr unsigned kQuantizeThreads = 4;

/// The geometry the wide route and the 320 x 10240 kernels are written for.
constexpr std::uint64_t kLowRank = 320;
constexpr std::uint64_t kHcDim = 10240;
constexpr std::uint64_t kHcCount = 4;

constexpr std::uint64_t Bf16Bytes(std::uint64_t rows, std::uint64_t cols) {
  return rows * cols * 2;
}
constexpr std::uint64_t Q8Bytes(std::uint64_t rows, std::uint64_t cols) {
  return rows * cols / 32 * 34;
}
constexpr std::uint64_t F32Bytes(std::uint64_t rows, std::uint64_t cols) {
  return rows * cols * 4;
}

struct Plan {
  /// True: narrow every listed BF16 down/up and widen every BF16 inject.
  bool convert{false};
  /// True when nothing is BF16 (a production-style package): the switch is a
  /// no-op and the log says so. False with `convert` false is a refusal.
  bool nothing_to_do{false};
  std::string reason;  ///< why `convert` is false
  std::size_t down{0}, up{0}, inject{0};
  std::array<std::size_t, kGroups> mixers_converted{};  ///< by Group
  std::uint64_t bf16_bytes{0};  ///< encoded bytes of the BF16 down/up
  std::uint64_t q8_bytes{0};    ///< the same tensors as Q8_0
  std::uint64_t inject_added_bytes{0};  ///< F32 minus BF16 for the inject
  [[nodiscard]] std::uint64_t resident_saved() const {
    return bf16_bytes - q8_bytes;
  }
};

/// All or nothing. Every mixer must end Q8_0 down and up, because the wide
/// mixer route is chosen once from layer 0 and a mixed model would hand some
/// kernel a BF16 pointer it was not written for. A mixer that already stores
/// Q8_0 (the shared MTP sidecar) counts as done. The output head and the
/// trunk's attention and FFN mixers must all be present.
inline Plan Decide(std::span<const Tensor> tensors) {
  Plan plan;
  const auto refuse = [&plan](std::string reason) {
    plan = Plan{};
    plan.reason = std::move(reason);
    return plan;
  };
  if (tensors.empty())
    return refuse("no hyper-connection tensors");
  std::array<std::size_t, kGroups> present{};
  std::uint32_t last_mixer = 0;
  bool have_last = false;
  core::GgmlType down_type = core::GgmlType::kF32;
  bool have_down = false;
  std::array<std::size_t, kGroups> bf16_mixers{};
  for (const Tensor& t : tensors) {
    const std::string who = std::string(Name(t.group)) + " mixer " +
                            std::to_string(t.mixer) + " " + Name(t.role);
    if (!have_last || t.mixer != last_mixer) {
      last_mixer = t.mixer;
      have_last = true;
      have_down = false;
      ++present[static_cast<std::size_t>(t.group)];
    }
    if (t.role == Role::kInject) {
      if (t.type != core::GgmlType::kF32 && t.type != core::GgmlType::kBF16 &&
          t.type != core::GgmlType::kQ8_0)
        return refuse(who + " type " + std::string(core::ToString(t.type)) +
                      " is not F32, BF16 or Q8_0");
      if (t.rows != kHcCount || t.cols != kHcDim)
        return refuse(who + " shape " + std::to_string(t.cols) + "x" +
                      std::to_string(t.rows) + " is not 10240x4");
      if (t.type == core::GgmlType::kBF16) {
        ++plan.inject;
        plan.inject_added_bytes += F32Bytes(t.rows, t.cols) - Bf16Bytes(t.rows, t.cols);
      }
      continue;
    }
    if (t.type != core::GgmlType::kBF16 && t.type != core::GgmlType::kQ8_0)
      return refuse(who + " type " + std::string(core::ToString(t.type)) +
                    " is not BF16 or Q8_0");
    const bool down = t.role == Role::kDown;
    if (down ? (t.rows != kLowRank || t.cols != kHcDim)
             : (t.rows != kHcDim || t.cols != kLowRank))
      return refuse(who + " shape " + std::to_string(t.cols) + "x" +
                    std::to_string(t.rows) + " is not " +
                    (down ? "10240x320" : "320x10240"));
    if (t.cols % 32 != 0)
      return refuse(who + " cols " + std::to_string(t.cols) +
                    " is not a multiple of 32");
    if (down) {
      down_type = t.type;
      have_down = true;
    } else if (!have_down || down_type != t.type) {
      return refuse(who + " type differs from its down projection");
    }
    if (t.type == core::GgmlType::kBF16) {
      ++(down ? plan.down : plan.up);
      plan.bf16_bytes += Bf16Bytes(t.rows, t.cols);
      plan.q8_bytes += Q8Bytes(t.rows, t.cols);
      if (!down)
        ++bf16_mixers[static_cast<std::size_t>(t.group)];
    }
  }
  if (present[static_cast<std::size_t>(Group::kTrunkAttn)] == 0 ||
      present[static_cast<std::size_t>(Group::kTrunkFfn)] == 0 ||
      present[static_cast<std::size_t>(Group::kOutputHead)] == 0)
    return refuse("trunk attention, trunk FFN or output mixer is missing");
  if (plan.down + plan.up == 0) {
    plan = Plan{};
    plan.nothing_to_do = true;
    plan.reason = "no BF16 hyper-connection projection (already Q8_0)";
    return plan;
  }
  plan.convert = true;
  plan.mixers_converted = bf16_mixers;
  return plan;
}

}  // namespace gufo::models::qwen38_flash_next::hc_load_q8
