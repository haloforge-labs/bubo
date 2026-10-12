// SPDX-License-Identifier: MIT
#include "src/npu/fabric_peak.hpp"

#include <Windows.h>

#include <cassert>
#include <iostream>
#include <stdexcept>

using namespace gufo::npu;
struct FakeClock final : ClockDriver {
  unsigned mode = 3, queries = 0, sets = 0, restores = 0;
  bool known = true, watcher = true, set_ok = true, restore_ok = true,
       throw_set = false;
  RestoreReason last = RestoreReason::Exit;
  ClockSnapshot Query() override {
    ++queries;
    return {known ? std::optional<unsigned>(mode) : std::nullopt,
            "unknown mode"};
  }
  bool IndependentRestorerReady() const noexcept override { return watcher; }
  bool SetPeak() override {
    ++sets;
    mode = 5;
    if (throw_set)
      throw std::runtime_error("setter threw after mutation");
    return set_ok;
  }
  bool Restore(unsigned previous, RestoreReason reason) noexcept override {
    ++restores;
    last = reason;
    if (!restore_ok)
      return false;
    mode = previous;
    return true;
  }
};
template<class F>
void Reject(F&& call) {
  bool rejected = false;
  try {
    call();
  } catch (const std::exception&) {
    rejected = true;
  }
  assert(rejected);
}
int main() {
  SetErrorMode(SEM_FAILCRITICALERRORS | SEM_NOGPFAULTERRORBOX);
  SetPriorityClass(GetCurrentProcess(), BELOW_NORMAL_PRIORITY_CLASS);
  auto token = gufo::core::ExperimentalAuthorization::FakeFabric();
  FakeClock denied;
  {
    FabricPeak policy({}, denied);
    Reject([&] { policy.Start(); });
  }
  assert(denied.queries == 0 && denied.sets == 0);
  {
    FabricPeak wrong_feature(
        gufo::core::ExperimentalAuthorization::FakePrefill(), denied);
    Reject([&] { wrong_feature.Start(); });
  }
  assert(denied.queries == 0 && denied.sets == 0);
  Reject([&] { InstalledOpenClClockDriver({}); });  // before even LoadLibrary
  FakeClock unknown;
  unknown.known = false;
  {
    FabricPeak policy(token, unknown);
    Reject([&] { policy.Start(); });
  }
  assert(unknown.queries == 1 && unknown.sets == 0 && unknown.restores == 0);
  FakeClock unwatched;
  unwatched.watcher = false;
  {
    FabricPeak policy(token, unwatched);
    Reject([&] { policy.Start(); });
  }
  assert(unwatched.sets == 0);
  for (auto reason :
       {RestoreReason::Exit, RestoreReason::Abort, RestoreReason::CrashHandler,
        RestoreReason::WatcherDeath}) {
    FakeClock clock;
    {
      FabricPeak policy(token, clock);
      policy.Start();
      assert(clock.mode == 5 && clock.queries == 1 && clock.sets == 1);
      assert(policy.Restore(reason));
      assert(policy.Restore(reason));
      assert(!policy.restore_pending());
    }
    assert(clock.mode == 3 && clock.restores == 1 && clock.last == reason);
  }
  FakeClock unwind;
  try {
    FabricPeak policy(token, unwind);
    policy.Start();
    throw std::runtime_error("abort");
  } catch (...) {
  }
  assert(unwind.mode == 3 && unwind.restores == 1);
  for (bool throwing : {false, true}) {
    FakeClock clock;
    clock.set_ok = false;
    clock.throw_set = throwing;
    {
      FabricPeak policy(token, clock);
      Reject([&] { policy.Start(); });
      assert(!policy.restore_pending());
      Reject([&] { policy.Start(); });
    }
    assert(clock.mode == 3 && clock.restores == 1 &&
           clock.last == RestoreReason::SetFailure);
  }
  FakeClock failed_restore;
  {
    FabricPeak policy(token, failed_restore);
    policy.Start();
    failed_restore.restore_ok = false;
    assert(!policy.Restore(RestoreReason::Abort) && policy.restore_pending());
    failed_restore.restore_ok = true;
    assert(policy.Restore(RestoreReason::WatcherDeath));
  }
  assert(failed_restore.mode == 3);
  std::cout
      << "PASS: deny before DLL/query/set; unknown prior mode and missing "
         "independent restorer deny mutation; exact previous mode restored on "
         "exit/abort/unwind/crash-handler/watcher-death/partial-set failure; "
         "failed restore remains pending\n";
}
