#include "src/core/platform/memory_guard.hpp"

#include <algorithm>
#include <cstdio>

namespace gufo::platform {
namespace {

constexpr std::uint64_t kMiB = std::uint64_t{1} << 20;

std::string Gib(std::uint64_t bytes) {
  char text[32];
  std::snprintf(text, sizeof(text), "%.1f", static_cast<double>(bytes) / static_cast<double>(kMiB * 1024));
  return text;
}

std::atomic<MemoryGuard*> g_process_guard{nullptr};

}  // namespace

AdmitResult Decide(const MemoryReadings& r, const MemoryNeed& need, std::uint64_t floor_bytes) {
  AdmitResult out;
  out.needed_bytes = need.mandatory_device_bytes + need.mandatory_host_bytes;
  if (floor_bytes == 0 || !r.ram_ok) return out;  // guard off, or no usable reading: do nothing
  const std::uint64_t ram_above_floor = r.ram_available > floor_bytes ? r.ram_available - floor_bytes : 0;
  out.free_bytes = ram_above_floor;
  if (out.needed_bytes == 0) return out;

  // Device-private bytes fill the free dedicated carve-out first; only the rest draws on system RAM (the shared slice of the local budget
  // IS system RAM and is already in the RAM reading, so it is not counted a second time).
  const std::uint64_t dev = need.mandatory_device_bytes;
  const std::uint64_t carve_free = r.gpu_carve_known ? std::min(r.gpu_carve_free, r.gpu_local_free) : r.gpu_local_free;
  const std::uint64_t dev_in_local = r.gpu_ok ? std::min(dev, carve_free) : 0;
  const std::uint64_t ram_need = need.mandatory_host_bytes + (dev - dev_in_local);
  std::uint64_t ram_short = ram_need > ram_above_floor ? ram_need - ram_above_floor : 0;
  if (r.commit_available != 0 && ram_need > r.commit_available)
    ram_short = std::max(ram_short, ram_need - r.commit_available);
  const std::uint64_t gpu_free = r.gpu_local_free + r.gpu_nonlocal_free;
  const std::uint64_t gpu_short = r.gpu_ok && dev > gpu_free ? dev - gpu_free : 0;
  if (ram_short == 0 && gpu_short == 0) return out;

  out.shortfall_bytes = std::max(ram_short, gpu_short);
  const bool ram_never = r.ram_total != 0 && ram_short != 0 &&
                         ram_need + floor_bytes > r.ram_total - std::min(r.process_ram_private, r.ram_total);
  const bool gpu_never = r.gpu_ok && r.gpu_budget_total != 0 && dev > r.gpu_budget_total;
  if (ram_never || gpu_never) {
    out.verdict = AdmitVerdict::kRefusePermanent;
    out.free_bytes = gpu_never ? gpu_free : ram_above_floor;
    out.message = "this request needs about " + Gib(out.needed_bytes) +
                  " GiB of memory, more than this computer can provide (" +
                  (gpu_never ? "GPU memory budget " + Gib(r.gpu_budget_total) + " GiB"
                             : "RAM " + Gib(r.ram_total) + " GiB, keep-free floor " + Gib(floor_bytes) + " GiB") +
                  "). Shorten the prompt or remove the image.";
    return out;
  }
  out.verdict = AdmitVerdict::kRefuseTransient;
  out.free_bytes = ram_short >= gpu_short ? ram_above_floor : gpu_free;
  out.message = "not enough free memory to start this request: it needs about " + Gib(out.needed_bytes) +
                " GiB; " + Gib(out.free_bytes) + " GiB is free above the " + Gib(floor_bytes) +
                " GiB keep-free floor. Free memory and retry.";
  return out;
}

StateFit PlanStateFit(const MemoryReadings& r, std::uint64_t floor_bytes, std::uint64_t reported_free_bytes) {
  StateFit fit;
  fit.capacity_bytes = reported_free_bytes;
  if (floor_bytes == 0 || !r.ram_ok || !r.gpu_carve_known)
    return fit;
  fit.known = true;
  fit.floor_bytes = floor_bytes;
  fit.carve_free_bytes = std::min(r.gpu_carve_free, r.gpu_local_free);
  fit.ram_available_bytes = r.ram_available;
  fit.ram_above_floor_bytes = r.ram_available > floor_bytes ? r.ram_available - floor_bytes : 0;
  fit.capacity_bytes = std::min(reported_free_bytes, fit.carve_free_bytes + r.ram_available);
  return fit;
}

StateSpill PlanStateSpill(const StateFit& fit, std::uint64_t need_bytes) {
  StateSpill spill;
  if (!fit.known)
    return spill;
  spill.known = true;
  spill.floor_bytes = fit.floor_bytes;
  spill.ram_available_bytes = fit.ram_available_bytes;
  spill.spill_bytes = need_bytes > fit.carve_free_bytes ? need_bytes - fit.carve_free_bytes : 0;
  spill.floor_eaten_bytes = spill.spill_bytes > fit.ram_above_floor_bytes ? spill.spill_bytes - fit.ram_above_floor_bytes : 0;
  return spill;
}

std::uint64_t StateCapacityBytes(std::uint64_t reported_free_bytes) {
  MemoryGuard* guard = ProcessMemoryGuard();
  if (guard == nullptr || !guard->enabled())
    return reported_free_bytes;
  return PlanStateFit(guard->Read(), guard->floor_bytes(), reported_free_bytes).capacity_bytes;
}

StateSpill StateSpillFor(std::uint64_t need_bytes) {
  MemoryGuard* guard = ProcessMemoryGuard();
  if (guard == nullptr || !guard->enabled())
    return {};
  return PlanStateSpill(PlanStateFit(guard->Read(), guard->floor_bytes(), ~std::uint64_t{0}), need_bytes);
}

const char* GuardLevelName(GuardLevel level) noexcept {
  switch (level) {
    case GuardLevel::kOk: return "ok";
    case GuardLevel::kSoft: return "soft";
    case GuardLevel::kLow: return "low";
    case GuardLevel::kCancelling: return "cancelling";
  }
  return "ok";
}

PressureStep PressureMachine::Step(const MemoryReadings& r, std::uint64_t floor_bytes,
                                   std::uint64_t retained_snapshot_bytes, Clock::time_point now) {
  PressureStep out;
  out.level = level_;
  if (floor_bytes == 0 || !r.ram_ok) return out;  // no usable reading: keep the level, do nothing

  const bool gpu_exhausted = r.gpu_ok && r.gpu_local_free + r.gpu_nonlocal_free == 0;
  const bool low_now = r.low_memory_signal || gpu_exhausted;
  const bool soft_now = r.ram_available < floor_bytes;
  soft_reads_ = soft_now ? soft_reads_ + 1 : 0;
  low_reads_ = low_now ? low_reads_ + 1 : 0;
  out.reason = low_now ? (r.low_memory_signal ? "low_memory_signal" : "gpu_budget") : soft_now ? "ram_below_floor" : "";

  if (!low_now) {
    if (!have_nolow_since_) { nolow_since_ = now; have_nolow_since_ = true; }
  } else {
    have_nolow_since_ = false;
  }
  if (!low_now && !soft_now) {
    if (!have_clear_since_) { clear_since_ = now; have_clear_since_ = true; }
  } else {
    have_clear_since_ = false;
  }

  const auto set_level = [&](GuardLevel level) {
    if (level == level_) return;
    level_ = level;
    ++changes_;
    out.level_changed = true;
  };

  // Escalation needs two consecutive readings, so a one-sample spike does not flap the level.
  if (low_reads_ >= 2 && level_ < GuardLevel::kLow) {
    set_level(GuardLevel::kLow);
    have_low_since_ = false;
  } else if (soft_reads_ >= 2 && level_ == GuardLevel::kOk) {
    set_level(GuardLevel::kSoft);
  }

  const auto action_due = [&] { return !have_action_time_ || now - last_action_ >= timing_.dwell; };
  if (level_ >= GuardLevel::kLow && low_now) {
    if (retained_snapshot_bytes > 0) {
      have_low_since_ = false;
      if (action_due()) {
        out.action.kind = PressureAction::Kind::kTrimSnapshots;
        out.action.trim_bytes = std::min(retained_snapshot_bytes, std::max(retained_snapshot_bytes / 4, 64 * kMiB));
        last_action_ = now;
        have_action_time_ = true;
      }
    } else {
      if (!have_low_since_) { low_since_ = now; have_low_since_ = true; }
      if (now - low_since_ >= timing_.cancel_after) {
        set_level(GuardLevel::kCancelling);
        if (action_due()) {
          out.action.kind = PressureAction::Kind::kCancelOne;
          last_action_ = now;
          have_action_time_ = true;
        }
      }
    }
  } else {
    have_low_since_ = false;
  }

  // De-escalation: only after the signal has stayed clear for the hold time.
  if (level_ >= GuardLevel::kLow && have_nolow_since_ && now - nolow_since_ >= timing_.recover_hold)
    set_level(soft_now ? GuardLevel::kSoft : GuardLevel::kOk);
  else if (level_ == GuardLevel::kSoft && have_clear_since_ && now - clear_since_ >= timing_.recover_hold)
    set_level(GuardLevel::kOk);

  out.level = level_;
  return out;
}

MemoryReadings MemoryGuard::Read() const noexcept {
  if (override_active_.load(std::memory_order_acquire)) {
    const std::lock_guard<std::mutex> lock(override_mutex_);
    return override_readings_;
  }
  return source_ != nullptr ? source_->Read() : MemoryReadings{};
}

void MemoryGuard::SetOverride(const MemoryReadings& readings) {
  {
    const std::lock_guard<std::mutex> lock(override_mutex_);
    override_readings_ = readings;
  }
  override_active_.store(true, std::memory_order_release);
}

void MemoryGuard::ClearOverride() { override_active_.store(false, std::memory_order_release); }

bool MemoryGuard::overridden() const { return override_active_.load(std::memory_order_acquire); }

AdmitResult MemoryGuard::Admit(const MemoryNeed& need, const std::function<std::uint64_t(std::uint64_t)>& trim) {
  if (!enabled()) return {};
  auto result = Decide(Read(), need, floor_bytes_);
  if (result.verdict == AdmitVerdict::kRefuseTransient && trim && result.shortfall_bytes != 0) {
    const std::uint64_t freed = trim(result.shortfall_bytes);
    if (freed != 0) {
      NoteTrimmed(freed);
      result = Decide(Read(), need, floor_bytes_);
    }
  }
  if (result.verdict == AdmitVerdict::kRefuseTransient) refused_transient_.fetch_add(1, std::memory_order_relaxed);
  if (result.verdict == AdmitVerdict::kRefusePermanent) refused_permanent_.fetch_add(1, std::memory_order_relaxed);
  return result;
}

MemoryGuard::Counters MemoryGuard::counters() const noexcept {
  return {refused_transient_.load(std::memory_order_relaxed), refused_permanent_.load(std::memory_order_relaxed),
          trimmed_.load(std::memory_order_relaxed), cancelled_.load(std::memory_order_relaxed)};
}

MemoryGuard* ProcessMemoryGuard() noexcept { return g_process_guard.load(std::memory_order_acquire); }
void SetProcessMemoryGuard(MemoryGuard* guard) noexcept { g_process_guard.store(guard, std::memory_order_release); }

}  // namespace gufo::platform
