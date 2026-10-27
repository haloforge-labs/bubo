// Memory guard decisions with fake OS readings (no OS, no GPU). A test imitates a small machine by choosing the numbers.
#include "src/core/platform/memory_guard.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <string_view>

namespace {

using namespace gufo::platform;
using std::chrono::milliseconds;
constexpr std::uint64_t GiB = std::uint64_t{1} << 30;
constexpr std::uint64_t MiB = std::uint64_t{1} << 20;
constexpr std::uint64_t kFloor = 4 * GiB;

void Expect(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Assertion failed: " << message << '\n';
    std::exit(1);
  }
}

// This machine: 31.6 GiB visible RAM, a 96 GiB GPU carve-out (the process's local budget), 13 GiB RAM free.
MemoryReadings Machine(std::uint64_t ram_available = 13 * GiB) {
  MemoryReadings r;
  r.ram_ok = true;
  r.ram_available = ram_available;
  r.ram_total = 31 * GiB;
  r.commit_available = 120 * GiB;
  r.process_ram_private = 3 * GiB;
  r.gpu_ok = true;
  r.gpu_local_free = 55 * GiB;
  r.gpu_nonlocal_free = 0;
  r.gpu_budget_total = 110 * GiB;
  return r;
}

MemoryNeed Vision() { return {.mandatory_device_bytes = 2 * GiB, .mandatory_host_bytes = 0, .optional_bytes = 3 * GiB}; }

struct FakeSource final : MemoryReadingSource {
  MemoryReadings readings;
  int reads{0};
  MemoryReadings Read() noexcept override { ++reads; return readings; }
};

void TestAdmitRules() {
  // Disabled floor and unusable readings admit anything.
  Expect(Decide(Machine(), Vision(), 0).verdict == AdmitVerdict::kAdmit, "floor 0 admits");
  MemoryReadings bad = Machine();
  bad.ram_ok = false;
  Expect(Decide(bad, {.mandatory_host_bytes = 500 * GiB}, kFloor).verdict == AdmitVerdict::kAdmit, "unusable reading admits");

  // A request with no mandatory bytes is served below the floor (snapshots are optional): the owner's game case.
  const auto text = Decide(Machine(3 * GiB), {.optional_bytes = 3 * GiB}, kFloor);
  Expect(text.verdict == AdmitVerdict::kAdmit, "text request below the floor is served");

  // The 96 GiB carve-out holds the device buffers, so 3 GiB of free RAM is no obstacle to a 2 GiB vision reserve.
  Expect(Decide(Machine(3 * GiB), Vision(), kFloor).verdict == AdmitVerdict::kAdmit, "vision fits in the local GPU budget");
  Expect(Decide(Machine(13 * GiB), Vision(), kFloor).verdict == AdmitVerdict::kAdmit, "normal image request is never refused at 13 GiB");

  // A small carve-out: only 1 GiB local is free, so 1 GiB of the 2 GiB lands in system RAM.
  MemoryReadings small = Machine(4 * GiB + 512 * MiB);
  small.gpu_local_free = 1 * GiB;
  small.gpu_nonlocal_free = 30 * GiB;
  const auto tight = Decide(small, Vision(), kFloor);
  Expect(tight.verdict == AdmitVerdict::kRefuseTransient, "1 GiB to RAM with 0.5 GiB above the floor is refused (transient)");
  Expect(tight.shortfall_bytes == 512 * MiB, "shortfall is what trimming snapshots must cover");
  Expect(tight.message.find("needs about 2.0 GiB") != std::string::npos, "message has the needed size");
  Expect(tight.message.find("0.5 GiB is free above the 4.0 GiB keep-free floor") != std::string::npos, "message has the free size and the floor");
  small.ram_available = 6 * GiB;
  Expect(Decide(small, Vision(), kFloor).verdict == AdmitVerdict::kAdmit, "enough RAM above the floor admits");

  // UMA double count: the local budget includes a ~15 GiB slice of system RAM beyond the 96 GiB carve-out. With the carve-out full, that
  // slice is already inside the RAM reading, so a 2 GiB device request draws on RAM and must respect the floor.
  MemoryReadings carve_full = Machine(5 * GiB);   // 1 GiB above the 4 GiB floor
  carve_full.gpu_local_free = 15 * GiB;           // all of it the shared slice
  carve_full.gpu_carve_known = true;
  carve_full.gpu_carve_free = 0;
  Expect(Decide(carve_full, Vision(), kFloor).verdict == AdmitVerdict::kRefuseTransient, "the shared slice of the local budget is not free GPU memory");
  carve_full.ram_available = 13 * GiB;
  Expect(Decide(carve_full, Vision(), kFloor).verdict == AdmitVerdict::kAdmit, "with 9 GiB of RAM above the floor the same request fits");
  carve_full.gpu_carve_known = false;   // unknown carve-out: the old reading (local free) is used
  carve_full.ram_available = 5 * GiB;
  Expect(Decide(carve_full, Vision(), kFloor).verdict == AdmitVerdict::kAdmit, "without carve-out data the local free is trusted");

  // GPU budget used up for a mandatory device request: transient.
  MemoryReadings gpu = Machine();
  gpu.gpu_local_free = 0;
  gpu.gpu_nonlocal_free = 1 * GiB;
  Expect(Decide(gpu, Vision(), kFloor).verdict == AdmitVerdict::kRefuseTransient, "GPU budget short is transient");

  // Can never fit: more than the machine can ever give.
  MemoryReadings tiny = Machine();
  tiny.ram_total = 16 * GiB;
  tiny.ram_available = 8 * GiB;
  tiny.process_ram_private = 2 * GiB;
  tiny.gpu_local_free = 0;
  tiny.gpu_nonlocal_free = 40 * GiB;
  const auto never = Decide(tiny, {.mandatory_host_bytes = 12 * GiB}, kFloor);
  Expect(never.verdict == AdmitVerdict::kRefusePermanent, "12 GiB + 4 GiB floor > 16 - 2 GiB is permanent");
  Expect(never.message.find("more than this computer can provide") != std::string::npos, "permanent message");
  const auto fits_if_idle = Decide(tiny, {.mandatory_host_bytes = 6 * GiB}, kFloor);
  Expect(fits_if_idle.verdict == AdmitVerdict::kRefuseTransient, "6 GiB does not fit now but could on an idle machine");
  MemoryReadings huge = Machine();
  Expect(Decide(huge, {.mandatory_device_bytes = 200 * GiB}, kFloor).verdict == AdmitVerdict::kRefusePermanent,
         "device bytes above the GPU budget total are permanent");

  // Commit headroom limits the charge too.
  MemoryReadings commit = Machine(20 * GiB);
  commit.gpu_local_free = 0;
  commit.commit_available = 1 * GiB;
  Expect(Decide(commit, {.mandatory_host_bytes = 2 * GiB}, kFloor).verdict == AdmitVerdict::kRefuseTransient, "commit headroom short");
}

void TestStateFit() {
  // This machine at load: 96 GiB carve-out with 24 GiB of weights in it, 13 GiB of RAM free; the process's reported free (110.9 GiB budget) counts
  // the 15 GiB shared slice that is also RAM.
  MemoryReadings r = Machine(13 * GiB);
  r.gpu_carve_known = true;
  r.gpu_carve_free = 72 * GiB;
  r.gpu_local_free = 87 * GiB;
  const auto fit = PlanStateFit(r, kFloor, 87 * GiB);
  Expect(fit.known && fit.capacity_bytes == 72 * GiB + 13 * GiB, "capacity = free carve-out + available RAM (floor ignored), not the reported 87 GiB");
  Expect(fit.ram_above_floor_bytes == 9 * GiB && fit.floor_bytes == kFloor, "the room above the floor is kept for the warning");
  r.ram_available = 2 * GiB;   // the game holds RAM: the start is still physically possible with the carve-out and that RAM
  Expect(PlanStateFit(r, kFloor, 87 * GiB).capacity_bytes == 74 * GiB, "RAM under the floor still counts toward the physical capacity");
  Expect(PlanStateFit(r, kFloor, 50 * GiB).capacity_bytes == 50 * GiB, "never more than the reported free");
  Expect(!PlanStateFit(r, 0, 87 * GiB).known && PlanStateFit(r, 0, 87 * GiB).capacity_bytes == 87 * GiB, "guard off: unchanged");
  MemoryReadings unknown = r;
  unknown.gpu_carve_known = false;
  Expect(!PlanStateFit(unknown, kFloor, 87 * GiB).known && PlanStateFit(unknown, kFloor, 87 * GiB).capacity_bytes == 87 * GiB, "no carve-out data: unchanged");
  // Through the process guard.
  FakeSource source;
  source.readings = r;
  source.readings.ram_available = 13 * GiB;
  MemoryGuard guard(&source, kFloor);
  Expect(StateCapacityBytes(87 * GiB) == 87 * GiB && !StateSpillFor(80 * GiB).known, "no process guard: unchanged, no spill");
  SetProcessMemoryGuard(&guard);
  Expect(StateCapacityBytes(87 * GiB) == 85 * GiB, "process guard: the physical capacity");
  Expect(StateSpillFor(70 * GiB).spill_bytes == 0 && StateSpillFor(75 * GiB).spill_bytes == 3 * GiB && StateSpillFor(75 * GiB).floor_eaten_bytes == 0,
         "spill = the part beyond the free carve-out; it fits above the floor");
  const auto eaten = StateSpillFor(84 * GiB);   // 12 GiB spill with 9 GiB above the floor
  Expect(eaten.known && eaten.spill_bytes == 12 * GiB && eaten.floor_eaten_bytes == 3 * GiB && eaten.floor_bytes == kFloor, "spill that eats into the floor says by how much");
  SetProcessMemoryGuard(nullptr);
}

void TestAdmitTrimsFirst() {
  FakeSource source;
  source.readings = Machine(4 * GiB + 100 * MiB);
  source.readings.gpu_local_free = 0;
  source.readings.gpu_nonlocal_free = 30 * GiB;
  MemoryGuard guard(&source, kFloor);
  const MemoryNeed need{.mandatory_device_bytes = 1 * GiB};
  int trim_calls = 0;
  const auto result = guard.Admit(need, [&](std::uint64_t shortfall) {
    ++trim_calls;
    Expect(shortfall > 0, "trim is asked for a positive shortfall");
    source.readings.ram_available += 2 * GiB;  // dropping snapshots returned memory
    return 2 * GiB;
  });
  Expect(trim_calls == 1, "trim called once");
  Expect(result.verdict == AdmitVerdict::kAdmit, "admitted after trimming");
  Expect(guard.counters().trimmed_bytes == 2 * GiB, "trimmed bytes counted");

  source.readings.ram_available = 4 * GiB + 100 * MiB;
  const auto refused = guard.Admit(need, [](std::uint64_t) { return std::uint64_t{0}; });
  Expect(refused.verdict == AdmitVerdict::kRefuseTransient, "nothing to trim: refused");
  Expect(guard.counters().refused_transient == 1, "refusal counted");

  MemoryGuard off(&source, 0);
  Expect(!off.enabled() && off.Admit(need).verdict == AdmitVerdict::kAdmit, "disabled guard admits");
}

void TestOverride() {
  FakeSource source;
  source.readings = Machine();
  MemoryGuard guard(&source, kFloor);
  Expect(guard.Read().ram_available == 13 * GiB, "real readings first");
  MemoryReadings fake = Machine(500 * MiB);
  guard.SetOverride(fake);
  Expect(guard.overridden() && guard.Read().ram_available == 500 * MiB, "override replaces the readings");
  guard.ClearOverride();
  Expect(!guard.overridden() && guard.Read().ram_available == 13 * GiB, "cleared override restores the readings");
}

PressureMachine::Clock::time_point At(int ms) {
  return PressureMachine::Clock::time_point{} + std::chrono::seconds(100) + milliseconds(ms);
}

void TestPressureMachine() {
  PressureMachine machine;
  const std::uint64_t snapshots = 8 * GiB;
  // One-sample spike does not move the level.
  MemoryReadings low = Machine(2 * GiB);
  auto step = machine.Step(low, kFloor, snapshots, At(0));
  Expect(step.level == GuardLevel::kOk && !step.level_changed, "single soft reading holds ok");
  step = machine.Step(Machine(), kFloor, snapshots, At(500));
  step = machine.Step(low, kFloor, snapshots, At(1000));
  Expect(step.level == GuardLevel::kOk, "a clear reading between resets the count");
  step = machine.Step(low, kFloor, snapshots, At(1500));
  Expect(step.level == GuardLevel::kSoft && step.level_changed, "two soft readings -> soft");
  Expect(std::string(step.reason) == "ram_below_floor", "soft reason");
  Expect(step.action.kind == PressureAction::Kind::kNone, "soft never trims (below the floor is the owner's normal)");
  // Hours below the floor: still soft, still no action.
  for (int i = 0; i < 100; ++i)
    step = machine.Step(low, kFloor, snapshots, At(2000 + 500 * i));
  Expect(step.level == GuardLevel::kSoft && step.action.kind == PressureAction::Kind::kNone, "steady below the floor stays soft, no action");

  // The Windows low-memory event: low, drop snapshots one step per dwell.
  int t = 60000;
  MemoryReadings event = Machine(1 * GiB);
  event.low_memory_signal = true;
  step = machine.Step(event, kFloor, snapshots, At(t));
  Expect(step.level == GuardLevel::kSoft, "one event reading is not enough");
  t += 500;
  step = machine.Step(event, kFloor, snapshots, At(t));
  Expect(step.level == GuardLevel::kLow && step.level_changed, "two event readings -> low");
  Expect(step.action.kind == PressureAction::Kind::kTrimSnapshots && step.action.trim_bytes == snapshots / 4, "first trim step is 25 %");
  Expect(std::string(step.reason) == "low_memory_signal", "low reason");
  t += 100;
  step = machine.Step(event, kFloor, snapshots, At(t));
  Expect(step.action.kind == PressureAction::Kind::kNone, "no second action inside the dwell");
  t += 500;
  step = machine.Step(event, kFloor, 40 * MiB, At(t));
  Expect(step.action.kind == PressureAction::Kind::kTrimSnapshots && step.action.trim_bytes == 40 * MiB, "last trim takes what is left");

  // Nothing left to drop and still low: cancel only after cancel_after, one request per dwell.
  t += 500;
  step = machine.Step(event, kFloor, 0, At(t));
  Expect(step.level == GuardLevel::kLow && step.action.kind == PressureAction::Kind::kNone, "no cancel yet");
  t += 1900;
  step = machine.Step(event, kFloor, 0, At(t));
  Expect(step.level == GuardLevel::kLow, "still inside the cancel wait");
  t += 200;
  step = machine.Step(event, kFloor, 0, At(t));
  Expect(step.level == GuardLevel::kCancelling && step.action.kind == PressureAction::Kind::kCancelOne, "cancel one request");
  t += 200;
  step = machine.Step(event, kFloor, 0, At(t));
  Expect(step.action.kind == PressureAction::Kind::kNone, "one cancel per dwell");
  t += 400;
  step = machine.Step(event, kFloor, 0, At(t));
  Expect(step.action.kind == PressureAction::Kind::kCancelOne, "next cancel after the dwell");

  // Recovery: the event clears, RAM still under the floor -> back to soft after the hold; then ok once above the floor.
  t += 500;
  step = machine.Step(Machine(3 * GiB), kFloor, 0, At(t));
  Expect(step.level == GuardLevel::kCancelling, "no step down before the hold");
  t += 5100;
  step = machine.Step(Machine(3 * GiB), kFloor, 0, At(t));
  Expect(step.level == GuardLevel::kSoft, "after the hold: soft while RAM is under the floor");
  t += 500;
  step = machine.Step(Machine(), kFloor, 0, At(t));
  t += 5100;
  step = machine.Step(Machine(), kFloor, 0, At(t));
  Expect(step.level == GuardLevel::kOk, "above the floor for the hold: ok");

  // GPU budget used up is also a low signal, with its own reason.
  PressureMachine gpu_machine;
  MemoryReadings gpu = Machine();
  gpu.gpu_local_free = 0;
  gpu.gpu_nonlocal_free = 0;
  gpu_machine.Step(gpu, kFloor, snapshots, At(0));
  step = gpu_machine.Step(gpu, kFloor, snapshots, At(500));
  Expect(step.level == GuardLevel::kLow && std::string(step.reason) == "gpu_budget", "GPU budget exhausted -> low");

  // No usable reading and a disabled floor change nothing.
  MemoryReadings bad;
  step = gpu_machine.Step(bad, kFloor, snapshots, At(10000));
  Expect(step.level == GuardLevel::kLow && step.action.kind == PressureAction::Kind::kNone, "unusable reading holds the level");
  PressureMachine off;
  off.Step(event, 0, snapshots, At(0));
  step = off.Step(event, 0, snapshots, At(500));
  Expect(step.level == GuardLevel::kOk, "floor 0: never leaves ok");
}

void TestProcessGuardPointer() {
  FakeSource source;
  MemoryGuard guard(&source, kFloor);
  Expect(guard.SnapshotsAllowed(), "snapshots allowed at ok");
  guard.SetLevel(GuardLevel::kSoft);
  Expect(guard.SnapshotsAllowed(), "below the floor alone the cache floor decides, not the guard");
  guard.SetLevel(GuardLevel::kLow);
  Expect(!guard.SnapshotsAllowed(), "snapshots stop at low");
  guard.SetLevel(GuardLevel::kCancelling);
  Expect(!guard.SnapshotsAllowed(), "snapshots stay off while cancelling");
  SetProcessMemoryGuard(&guard);
  Expect(ProcessMemoryGuard() == &guard, "process guard set");
  SetProcessMemoryGuard(nullptr);
  Expect(ProcessMemoryGuard() == nullptr, "process guard cleared");
}

}  // namespace

int main() {
  TestAdmitRules();
  TestStateFit();
  TestAdmitTrimsFirst();
  TestOverride();
  TestPressureMachine();
  TestProcessGuardPointer();
  std::cout << "memory_guard_test OK\n";
  return 0;
}
