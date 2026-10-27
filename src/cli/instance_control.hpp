#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/json.hpp"

namespace bubo::cli {
using gufo::json::Value;
inline constexpr int kControlVersion = 1;
inline constexpr std::uint32_t kControlDeadlineMs = 2000;

class ControlError : public std::runtime_error {
public:
  ControlError(int exit_code, const std::string& message);
  /// `details` become fields of the error payload (memory_insufficient: needed_bytes, available_bytes, sessions, context, per_session_bytes, floor_bytes).
  ControlError(int exit_code, const std::string& message, Value details);
  int ExitCode() const { return exit_code_; }
  const Value& Details() const { return details_; }

private:
  int exit_code_;
  Value details_;
};

struct InstanceIdentity {
  std::string instance_id;
  std::uint32_t pid = 0;
  // Exact Windows FILETIME ticks, encoded as canonical decimal text.
  std::string process_creation_time;
  std::string exe_path;
};
InstanceIdentity CurrentIdentity(std::string instance_id);
Value IdentityJson(const InstanceIdentity& identity);
bool ValidInstanceId(std::string_view instance_id);
std::string ControlPipeName(std::string_view instance_id);

struct DiscoveryRecord {
  InstanceIdentity identity;
  std::string endpoint;
  std::string phase;
  std::optional<std::string> log_path;
  Value last_error;
};
// root is the LOCALAPPDATA directory, not the instances directory. An empty
// root uses the current user's LOCALAPPDATA. No recursive filesystem search.
std::filesystem::path DiscoveryDirectory(std::string_view root = {});
Value DiscoveryJson(const DiscoveryRecord& record);
void WriteDiscovery(const DiscoveryRecord& record, std::string_view root = {});
void WriteTombstone(const DiscoveryRecord& record, const Value& last_error = {},
                    std::string_view root = {});
std::optional<DiscoveryRecord> ReadDiscovery(std::string_view instance_id,
                                             std::string_view root = {});
std::vector<DiscoveryRecord> DiscoverRecords(std::string_view root = {});
// Only an owner-secured, stopped record whose exact process has exited is
// accepted. A live process (including PID reuse) never counts as stopped.
bool RecordedStopped(const DiscoveryRecord& record);

class ControlServer {
public:
  ControlServer(InstanceIdentity identity, std::function<Value()> snapshot,
                std::function<void()> stop);
  ~ControlServer();
  ControlServer(const ControlServer&) = delete;
  ControlServer& operator=(const ControlServer&) = delete;
  void Start();
  void Shutdown();

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

class VerifiedInstance {
public:
  ~VerifiedInstance();
  VerifiedInstance(VerifiedInstance&&) noexcept;
  VerifiedInstance& operator=(VerifiedInstance&&) noexcept;
  VerifiedInstance(const VerifiedInstance&) = delete;
  VerifiedInstance& operator=(const VerifiedInstance&) = delete;
  const InstanceIdentity& Identity() const;
  Value Snapshot();
  Value RequestStop();
  bool WaitForExit(std::uint32_t wait_ms);

private:
  class Impl;
  explicit VerifiedInstance(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
  friend VerifiedInstance VerifiedConnect(const DiscoveryRecord&,
                                          std::uint32_t);
};
VerifiedInstance VerifiedConnect(const DiscoveryRecord& record,
                                 std::uint32_t timeout_ms = kControlDeadlineMs);

// Acquire fails with port_occupied/13 if another CLI owns this endpoint. The
// lease stays held through server shutdown, including weight cleanup.
class EndpointLease {
public:
  EndpointLease(std::string host, int port);
  ~EndpointLease();
  EndpointLease(const EndpointLease&) = delete;
  EndpointLease& operator=(const EndpointLease&) = delete;
  void Acquire();

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

// Reserve before weight loading; release immediately before HttpServer bind.
// The actual bind remains authoritative and must still report port_occupied.
class PortReservation {
public:
  PortReservation(std::string host, int port);
  ~PortReservation();
  PortReservation(const PortReservation&) = delete;
  PortReservation& operator=(const PortReservation&) = delete;
  void Reserve();
  int Port() const;
  void Release();

private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};
}  // namespace bubo::cli
