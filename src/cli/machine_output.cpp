#include "src/cli/machine_output.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <ctime>
#include <iostream>
#include <random>
#include <stdexcept>
#include <streambuf>

#include "src/cli/arg_parser.hpp"
#include "src/cli/settings_support.hpp"

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#include <windows.h>
#include <winternl.h>
#else
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <unistd.h>

#include <cerrno>
#include <csignal>
#endif

#include "src/cli/brand.hpp"

namespace bubo::cli {
namespace {
#ifdef _WIN32
std::intptr_t gBootstrapOutput = -1;
#endif
int Duplicate(int fd) {
#ifdef _WIN32
  return _dup(fd);
#else
  return dup(fd);
#endif
}
int Redirect(int from, int to) {
#ifdef _WIN32
  return _dup2(from, to);
#else
  return dup2(from, to) < 0 ? -1 : 0;
#endif
}
void Close(int fd) {
#ifdef _WIN32
  _close(fd);
#else
  close(fd);
#endif
}
std::string NewId() {
  std::random_device random;
  constexpr char hex[] = "0123456789abcdef";
  std::string result;
  for (int i = 0; i < 16; ++i) {
    const auto byte = random() & 255;
    result += hex[byte >> 4];
    result += hex[byte & 15];
  }
  return result;
}
}  // namespace

class MachineWriter::NativeOutput : public std::streambuf {
public:
  NativeOutput() {
#ifdef _WIN32
    if (gBootstrapOutput != -1) {
      borrowed_ = gBootstrapOutput;
      return;
    }
#endif
    std::cout.flush();
    std::fflush(stdout);
    descriptor_ = Duplicate(1);
    if (descriptor_ < 0 || Redirect(2, 1) != 0) {
      if (descriptor_ >= 0)
        Close(descriptor_);
      throw std::runtime_error("cannot isolate machine stdout");
    }
#ifdef _WIN32
    _setmode(descriptor_, _O_BINARY);
    // The wire handle belongs only to this process. Children inherit the
    // redirected stderr handle, never the saved machine stdout pipe.
    if (!SetHandleInformation(reinterpret_cast<HANDLE>(Handle()),
                              HANDLE_FLAG_INHERIT, 0)) {
      Redirect(descriptor_, 1);
      Close(descriptor_);
      descriptor_ = -1;
      throw std::runtime_error("cannot protect machine stdout handle");
    }
    SetStdHandle(STD_OUTPUT_HANDLE,
                 reinterpret_cast<HANDLE>(_get_osfhandle(1)));
#else
    if (fcntl(descriptor_, F_SETFD, FD_CLOEXEC) < 0) {
      Redirect(descriptor_, 1);
      Close(descriptor_);
      descriptor_ = -1;
      throw std::runtime_error("cannot protect machine stdout descriptor");
    }
#endif
  }
  ~NativeOutput() override {
    if (descriptor_ < 0)
      return;
    std::cout.flush();
    std::fflush(stdout);
    if (restore_)
      Redirect(descriptor_, 1);
#ifdef _WIN32
    SetStdHandle(STD_OUTPUT_HANDLE,
                 reinterpret_cast<HANDLE>(_get_osfhandle(1)));
#endif
    Close(descriptor_);
  }
  void KeepRedirected() { restore_ = false; }
  bool Connected() const {
#ifdef _WIN32
    const auto handle = reinterpret_cast<HANDLE>(Handle());
    if (GetFileType(handle) != FILE_TYPE_PIPE)
      return true;
    // A zero-byte write need not detect a closed pipe. Query the local pipe
    // state without writing a heartbeat or blocking on the reader.
    struct LocalPipeInfo {
      ULONG type, configuration, maximum_instances, current_instances;
      ULONG inbound_quota, read_available, outbound_quota, write_available;
      ULONG state, end;
    };
    using Query = NTSTATUS(NTAPI*)(HANDLE, PIO_STATUS_BLOCK, PVOID, ULONG,
                                   FILE_INFORMATION_CLASS);
    static const auto query = reinterpret_cast<Query>(GetProcAddress(
        GetModuleHandleW(L"ntdll.dll"), "NtQueryInformationFile"));
    IO_STATUS_BLOCK status{};
    LocalPipeInfo info{};
    // FilePipeLocalInformation = 24; FILE_PIPE_CONNECTED_STATE = 3.
    return query &&
           query(handle, &status, &info, sizeof(info),
                 static_cast<FILE_INFORMATION_CLASS>(24)) >= 0 &&
           info.state == 3;
#else
    pollfd descriptor{descriptor_, POLLOUT, 0};
    return poll(&descriptor, 1, 0) >= 0 &&
           !(descriptor.revents & (POLLERR | POLLHUP | POLLNVAL));
#endif
  }
  bool Send(std::string_view line) {
#ifdef _WIN32
    // Anonymous parent pipes can block in WriteFile. A worker owns its bytes
    // and a duplicate handle, so timeout never leaves references to this
    // writer or its caller. Cancel only this worker's synchronous write.
    auto task = std::make_shared<WriteTask>();
    task->line = std::string(line);
    if (!DuplicateHandle(GetCurrentProcess(),
                         reinterpret_cast<HANDLE>(Handle()),
                         GetCurrentProcess(), &task->output, 0, FALSE,
                         DUPLICATE_SAME_ACCESS))
      return false;
    auto* owned = new std::shared_ptr<WriteTask>(task);
    const auto worker = CreateThread(nullptr, 0, SendTask, owned, 0, nullptr);
    if (!worker) {
      delete owned;
      return false;
    }
    const bool finished = WaitForSingleObject(worker, 2000) == WAIT_OBJECT_0;
    if (!finished) {
      task->cancelled.store(true);
      for (unsigned retry = 0; retry < 20; ++retry) {
        CancelSynchronousIo(worker);
        if (WaitForSingleObject(worker, 5) == WAIT_OBJECT_0)
          break;
      }
    }
    CloseHandle(worker);
    return finished && task->success.load();
#else
    // Suppress SIGPIPE only on this thread, and consume a new pending signal
    // before restoring its mask. A closed consumer must return a write error.
    sigset_t blocked, previous, pending;
    sigemptyset(&blocked);
    sigaddset(&blocked, SIGPIPE);
    pthread_sigmask(SIG_BLOCK, &blocked, &previous);
    sigpending(&pending);
    const bool already_pending = sigismember(&pending, SIGPIPE);
    const auto deadline =
        std::chrono::steady_clock::now() + std::chrono::seconds(2);
    bool success = true;
    while (!line.empty()) {
      const auto left = std::chrono::duration_cast<std::chrono::milliseconds>(
                            deadline - std::chrono::steady_clock::now())
                            .count();
      pollfd descriptor{descriptor_, POLLOUT, 0};
      if (left <= 0 || poll(&descriptor, 1, static_cast<int>(left)) <= 0 ||
          !(descriptor.revents & POLLOUT)) {
        success = false;
        break;
      }
      const auto sent = write(descriptor_, line.data(),
                              std::min<std::size_t>(line.size(), 4096));
      if (sent <= 0) {
        if (sent < 0 && errno == EINTR)
          continue;
        success = false;
        break;
      }
      line.remove_prefix(static_cast<std::size_t>(sent));
    }
    if (!already_pending) {
      timespec zero{};
      sigtimedwait(&blocked, nullptr, &zero);
    }
    pthread_sigmask(SIG_SETMASK, &previous, nullptr);
    return success;
#endif
  }
  std::intptr_t Handle() const {
#ifdef _WIN32
    return borrowed_ != -1 ? borrowed_ : _get_osfhandle(descriptor_);
#else
    return descriptor_;
#endif
  }

private:
#ifdef _WIN32
  struct WriteTask {
    HANDLE output = nullptr;
    std::string line;
    std::atomic<bool> cancelled{false}, success{false};
    ~WriteTask() {
      if (output)
        CloseHandle(output);
    }
  };
  static DWORD WINAPI SendTask(void* context) {
    std::unique_ptr<std::shared_ptr<WriteTask>> owner(
        static_cast<std::shared_ptr<WriteTask>*>(context));
    const auto task = *owner;
    std::string_view remaining(task->line);
    while (!remaining.empty() && !task->cancelled.load()) {
      DWORD sent = 0;
      if (!WriteFile(task->output, remaining.data(),
                     static_cast<DWORD>(
                         std::min<std::size_t>(remaining.size(), 65536)),
                     &sent, nullptr) ||
          !sent)
        return 0;
      remaining.remove_prefix(sent);
    }
    task->success.store(remaining.empty());
    return 0;
  }
#endif
  int descriptor_ = -1;
  std::intptr_t borrowed_ = -1;
  bool restore_ = true;
};

void ConfigureMachineOutputHandle(std::intptr_t handle) {
#ifdef _WIN32
  gBootstrapOutput = handle;
#else
  (void)handle;
#endif
}

bool MachineRequested(std::span<const char* const> args) {
  return gufo::cli::ArgParser::Contains(args, {"--json", "--jsonl"});
}
MachineMode ParseMachineMode(std::span<const char* const> args) {
  MachineMode result = MachineMode::Human;
  gufo::cli::ArgParser parser("machine output");
  parser.SetImplicitHelp(false).SetInlineValues(false);
  parser.SetUnrecognizedHandler([](auto, auto*) { return true; });
  const auto select = [&](std::string_view flag, auto, auto*) {
    if (result != MachineMode::Human)
      throw std::invalid_argument("select one machine-output flag");
    result = flag == "--json" ? MachineMode::Json : MachineMode::JsonLines;
    return true;
  };
  parser.AddCustomFlag("", "--json", "Machine output", "Output", select);
  parser.AddCustomFlag("", "--jsonl", "Machine stream", "Output", select);
  // This routing pass deliberately observes every raw token, including value
  // positions; the command's parser owns value consumption and validation.
  (void)parser.Parse(args);
  return result;
}
Value Product() {
  auto value = Value::object();
  value["name"] = std::string(gufo::cli::kProductName);
  value["version"] = std::string(gufo::cli::kProductVersion);
  value["revision"] = std::string(gufo::cli::kRevision);
  value["engine_version"] = std::string(gufo::cli::kEngineVersion);
  return value;
}
Value Capabilities() {
  auto value = Value::object();
  value["schema_version"] = 1;
  value["prepare_jsonl"] = true;
  value["diagnose_json"] = true;
  value["status_json"] = true;
#ifdef _WIN32
  value["instance_control"] = true;
#else
  value["instance_control"] = false;
#endif
  value["status_watch"] = true;
  value["settings"] = true;
  value["profiles"] = SettingsProfiles();
  value["memory_estimate"] = true;
  value["connection_helpers"] = true;
  value["support_bundle"] = true;
  value["switch_registry"] = true;
  value["request_history"] = true;
  value["per_slot_context"] = false;
  value["machine_readings"] = true;
  return value;
}
Value Error(std::string_view code, std::string_view component,
            std::string_view stage, bool retryable, std::string_view remedy,
            std::string_view detail) {
  auto value = Value::object();
  value["code"] = std::string(code);
  value["component"] = std::string(component);
  value["stage"] = std::string(stage);
  value["retryable"] = retryable;
  value["remedy"] = std::string(remedy);
  if (detail.size() > 4096) {
    // At most 4 KiB, cut before a UTF-8 continuation byte.
    std::size_t end = 4096;
    while (end && (static_cast<unsigned char>(detail[end]) & 0xc0) == 0x80)
      --end;
    detail = detail.substr(0, end);
  }
  if (!detail.empty())
    value["detail"] = std::string(detail);
  return value;
}
LastLineCapture::LastLineCapture(std::streambuf* target, std::string prefix)
    : target_(target), prefix_(std::move(prefix)) {}
std::string LastLineCapture::Last() const {
  std::lock_guard lock(mutex_);
  return last_;
}
void LastLineCapture::Reset() {
  std::lock_guard lock(mutex_);
  last_.clear();
}
LastLineCapture::int_type LastLineCapture::overflow(int_type ch) {
  if (traits_type::eq_int_type(ch, traits_type::eof()))
    return traits_type::not_eof(ch);
  const char byte = traits_type::to_char_type(ch);
  Observe(&byte, 1);
  return target_ ? target_->sputc(byte) : ch;
}
std::streamsize LastLineCapture::xsputn(const char* text,
                                        std::streamsize count) {
  if (count > 0)
    Observe(text, static_cast<std::size_t>(count));
  return target_ ? target_->sputn(text, count) : count;
}
int LastLineCapture::sync() {
  return target_ ? target_->pubsync() : 0;
}
void LastLineCapture::Observe(const char* text, std::size_t count) {
  constexpr std::size_t kLimit = 4096;
  std::lock_guard lock(mutex_);
  for (std::size_t i = 0; i < count; ++i) {
    if (text[i] != '\n') {
      if (current_.size() < kLimit)
        current_ += text[i];
      continue;
    }
    if (!current_.empty() && current_.back() == '\r')
      current_.pop_back();
    // Drop a UTF-8 sequence cut by the limit.
    auto end = current_.size();
    while (end &&
           (static_cast<unsigned char>(current_[end - 1]) & 0xc0) == 0x80)
      --end;
    if (end && (static_cast<unsigned char>(current_[end - 1]) & 0x80)) {
      const auto lead = static_cast<unsigned char>(current_[end - 1]);
      const std::size_t length = lead >= 0xf0   ? 4
                                 : lead >= 0xe0 ? 3
                                 : lead >= 0xc0 ? 2
                                                : 1;
      if (current_.size() - (end - 1) < length)
        current_.resize(end - 1);
    }
    if (current_.starts_with(prefix_))
      last_ = current_;
    current_.clear();
  }
}
Value PrepareError(int code, std::string_view stage, std::string_view detail) {
  constexpr std::array<const char*, 11> names = {
      "success",          "unexpected_failure", "invalid_arguments",
      "cancelled",        "source_missing",     "transport_failed",
      "integrity_failed", "storage_failed",     "prepare_locked",
      "prepare_internal", "unsupported"};
  constexpr std::array<const char*, 11> remedies = {
      "",
      "Run prepare again.",
      "Run bubo prepare --help.",
      "Run prepare and approve the plan.",
      "Restore the pinned source files.",
      "Check the connection. Run prepare again.",
      "Remove the invalid file. Run prepare again.",
      "Check folder access and free space.",
      "Wait for the other prepare command.",
      "Run prepare again. If it fails again, read the detail.",
      "Use a supported command."};
  const auto index = code >= 0 && code <= 10 ? code : 9;
  return Error(names[index], "prepare", stage, code == 5 || code == 8,
               remedies[index], detail);
}
std::string UtcTime() {
  const auto now = std::chrono::system_clock::now();
  const auto seconds = std::chrono::system_clock::to_time_t(now);
  std::tm utc{};
#ifdef _WIN32
  gmtime_s(&utc, &seconds);
#else
  gmtime_r(&seconds, &utc);
#endif
  char text[32] = {};
  std::strftime(text, sizeof(text), "%Y-%m-%dT%H:%M:%SZ", &utc);
  return text;
}
MachineWriter::MachineWriter(std::ostream& output, bool keep_stdout_redirected)
    : output_(output), operation_id_(NewId()) {
  if (&output == &std::cout) {
    static std::mutex native_mutex;
    static std::weak_ptr<NativeOutput> active;
    std::lock_guard lock(native_mutex);
    native_ = active.lock();
    if (!native_) {
      native_ = std::make_shared<NativeOutput>();
      active = native_;
    }
    if (keep_stdout_redirected)
      native_->KeepRedirected();
  }
}
MachineWriter::~MachineWriter() = default;
std::intptr_t MachineWriter::NativeHandle() const {
  return native_ ? native_->Handle() : -1;
}
bool MachineWriter::Connected() {
  std::lock_guard lock(mutex_);
  if (!failed_)
    failed_ = native_ ? !native_->Connected() : !output_;
  return !failed_;
}
bool MachineWriter::Write(std::string_view type, Value payload,
                          const Value* legacy_fields) {
  std::lock_guard lock(mutex_);
  if (failed_)
    return false;
  auto record = legacy_fields ? *legacy_fields : Value::object();
  record["schema_version"] = 1;
  record["type"] = std::string(type);
  record["operation_id"] = operation_id_;
  record["seq"] = seq_ + 1;
  record["time"] = UtcTime();
  record["product"] = Product();
  record["payload"] = std::move(payload);
  auto line = record.dump();
  if (line.size() >= 1024 * 1024)
    throw std::length_error("machine event exceeds 1 MiB");
  ++seq_;
  line += '\n';
  if (native_)
    failed_ = !native_->Send(line);
  else {
    output_ << line << std::flush;
    failed_ = !output_;
  }
  return !failed_;
}
}  // namespace bubo::cli
