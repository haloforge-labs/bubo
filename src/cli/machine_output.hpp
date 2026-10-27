#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <ostream>
#include <span>
#include <streambuf>
#include <string>
#include <string_view>

#include "src/core/json.hpp"

namespace bubo::cli {
using gufo::json::Value;
enum class MachineMode { Human, Json, JsonLines };
bool MachineRequested(std::span<const char* const> args);
MachineMode ParseMachineMode(std::span<const char* const> args);
Value Capabilities();
Value Product();
// `detail` is the cause in the command's own words (the text it also writes
// to stderr). The field is omitted when there is none.
Value Error(std::string_view code, std::string_view component,
            std::string_view stage, bool retryable, std::string_view remedy,
            std::string_view detail = {});
Value PrepareError(int exit_code, std::string_view stage,
                   std::string_view detail = {});

// Forwards every byte to `target` and keeps the last complete line that
// starts with `prefix` (at most 4 KiB, cut at a UTF-8 boundary). Machine
// errors take their `detail` from it. Safe for writers on several threads.
class LastLineCapture : public std::streambuf {
public:
  LastLineCapture(std::streambuf* target, std::string prefix);
  std::string Last() const;
  void Reset();

protected:
  int_type overflow(int_type ch) override;
  std::streamsize xsputn(const char* text, std::streamsize count) override;
  int sync() override;

private:
  void Observe(const char* text, std::size_t count);
  std::streambuf* target_;
  std::string prefix_, current_, last_;
  mutable std::mutex mutex_;
};
std::string UtcTime();
// Bootstrap passes the saved OS stdout handle across the engine DLL boundary.
// The handle remains owned by its outer writer for the command's lifetime.
void ConfigureMachineOutputHandle(std::intptr_t handle);

// The writer owns stdout in machine mode. Native library writes to fd 1 and
// std::cout go to stderr; only the saved descriptor carries wire records.
// Custom streams used by CPU tests do not alter process descriptors.
class MachineWriter {
public:
  explicit MachineWriter(std::ostream& output,
                         bool keep_stdout_redirected = false);
  ~MachineWriter();
  MachineWriter(const MachineWriter&) = delete;
  MachineWriter& operator=(const MachineWriter&) = delete;
  bool Write(std::string_view type, Value payload,
             const Value* legacy_fields = nullptr);
  const std::string& OperationId() const { return operation_id_; }
  std::intptr_t NativeHandle() const;
  // Poll while waiting for stdin as well as during active output. A consumer
  // can close stdout after the last complete record, with no write pending.
  bool Connected();

private:
  class NativeOutput;
  std::shared_ptr<NativeOutput> native_;
  std::ostream& output_;
  std::string operation_id_;
  std::mutex mutex_;
  std::uint64_t seq_ = 0;
  bool failed_ = false;
};
}  // namespace bubo::cli
