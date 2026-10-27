#pragma once

namespace bubo::cli {
// These values apply only to machine-mode lifecycle/control commands.
// Prepare, diagnose and human serve retain their command-scoped exit values.
enum class MachineExit {
  Success = 0,
  Unexpected = 1,
  InvalidArguments = 2,
  RuntimeMissing = 11,
  ModelMissing = 12,
  PortOccupied = 13,
  StartupTimeout = 14,
  ProtocolMismatch = 15,
  ControlAccess = 16,
  StopTimeout = 17,
  Unsupported = 18,
  MemoryInsufficient = 19,
  DeviceLost = 75
};
constexpr int LifecycleExit(bool machine, MachineExit code, int legacy) {
  return machine ? static_cast<int>(code) : legacy;
}
}  // namespace bubo::cli
