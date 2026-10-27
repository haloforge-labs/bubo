#include "src/cli/instance_control.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <limits>
#include <thread>
#include <utility>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <aclapi.h>
#include <sddl.h>
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#endif

#include "src/cli/machine_output.hpp"
#include "src/cli/prepared_default.hpp"
#include "src/core/prepare/metadata_json.hpp"

namespace bubo::cli {
namespace {
namespace fs = std::filesystem;
namespace metadata = bubo::prepare::metadata;
constexpr std::size_t kRequestBytes = 65536;
constexpr std::size_t kRecordBytes = 1024 * 1024;
[[maybe_unused, noreturn]] void Unsupported() {
  throw ControlError(18, "local instance control requires Windows");
}

bool Hex(std::string_view text, std::size_t length) {
  return text.size() == length &&
         std::all_of(text.begin(), text.end(), [](char c) {
           return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
         });
}
bool CreationTicks(std::string_view text) {
  if (text.empty() || text.front() == '0')
    return false;
  std::uint64_t ticks = 0;
  const auto [end, error] =
      std::from_chars(text.data(), text.data() + text.size(), ticks);
  return error == std::errc() && end == text.data() + text.size() && ticks != 0;
}

InstanceIdentity ParseIdentity(const Value& value) {
  metadata::Keys(value,
                 {"instance_id", "pid", "process_creation_time", "exe_path"});
  InstanceIdentity identity;
  identity.instance_id =
      metadata::String(metadata::Required(value, "instance_id"));
  const auto pid = metadata::Integer(metadata::Required(value, "pid"));
  if (pid == 0 || pid > std::numeric_limits<std::uint32_t>::max())
    throw ControlError(16, "invalid process identity");
  identity.pid = static_cast<std::uint32_t>(pid);
  identity.process_creation_time =
      metadata::String(metadata::Required(value, "process_creation_time"));
  identity.exe_path = metadata::String(metadata::Required(value, "exe_path"));
  if (!ValidInstanceId(identity.instance_id) ||
      !CreationTicks(identity.process_creation_time) ||
      identity.exe_path.empty() ||
      identity.exe_path.find('\0') != std::string::npos)
    throw ControlError(16, "invalid process identity");
  return identity;
}

[[maybe_unused]] void CheckVersions(const Value& value) {
  if (metadata::Integer(metadata::Required(value, "schema_version")) != 1 ||
      metadata::Integer(metadata::Required(value, "control_version")) !=
          kControlVersion)
    throw ControlError(15, "unsupported local control version");
}

[[maybe_unused]] Value Protocol(std::string_view action,
                                const InstanceIdentity& identity) {
  auto value = Value::object();
  value["schema_version"] = 1;
  value["control_version"] = kControlVersion;
  value["action"] = std::string(action);
  value["identity"] = IdentityJson(identity);
  return value;
}

Value RedactedError(const Value& error) {
  if (error.is_null())
    return {};
  // Never persist free-form messages, paths, prompts, keys or request data.
  // Unknown error fields are discarded; the remedy is a fixed safe string.
  auto result = Value::object();
  for (const auto* name : {"code", "component", "stage"}) {
    const auto text = error.member_str(name);
    if (text.empty() || text.size() > 80 ||
        !std::all_of(text.begin(), text.end(), [](char c) {
          return (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '_' ||
                 c == '-';
        }))
      result[name] = "unknown";
    else
      result[name] = text;
  }
  const auto* retryable = error.find("retryable");
  result["retryable"] = retryable && retryable->as_bool();
  result["remedy"] = "Run bubo status --json. Read the local log if available.";
  return result;
}

#ifdef _WIN32
class Handle {
public:
  explicit Handle(HANDLE value = nullptr) : value_(value) {}
  ~Handle() { Reset(); }
  Handle(const Handle&) = delete;
  Handle& operator=(const Handle&) = delete;
  Handle(Handle&& other) noexcept : value_(other.Release()) {}
  Handle& operator=(Handle&& other) noexcept {
    if (this != &other)
      Reset(other.Release());
    return *this;
  }
  HANDLE Get() const { return value_; }
  explicit operator bool() const {
    return value_ && value_ != INVALID_HANDLE_VALUE;
  }
  HANDLE Release() { return std::exchange(value_, nullptr); }
  void Reset(HANDLE value = nullptr) {
    if (*this)
      CloseHandle(value_);
    value_ = value;
  }

private:
  HANDLE value_;
};

std::wstring Wide(std::string_view text) {
  if (text.size() > static_cast<std::size_t>(std::numeric_limits<int>::max()))
    throw ControlError(16, "control path is too long");
  const int count =
      MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(),
                          static_cast<int>(text.size()), nullptr, 0);
  if (count == 0 && !text.empty())
    throw ControlError(16, "invalid UTF-8 control path");
  std::wstring result(static_cast<std::size_t>(count), L'\0');
  if (count)
    MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, text.data(),
                        static_cast<int>(text.size()), result.data(), count);
  return result;
}

std::string Utf8(std::wstring_view text) {
  const int count = WideCharToMultiByte(
      CP_UTF8, WC_ERR_INVALID_CHARS, text.data(), static_cast<int>(text.size()),
      nullptr, 0, nullptr, nullptr);
  if (count == 0 && !text.empty())
    throw ControlError(16, "invalid Windows control path");
  std::string result(static_cast<std::size_t>(count), '\0');
  if (count)
    WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, text.data(),
                        static_cast<int>(text.size()), result.data(), count,
                        nullptr, nullptr);
  return result;
}

std::vector<unsigned char> TokenSid(HANDLE token) {
  DWORD bytes = 0;
  GetTokenInformation(token, TokenUser, nullptr, 0, &bytes);
  if (GetLastError() != ERROR_INSUFFICIENT_BUFFER || bytes == 0)
    throw ControlError(16, "cannot query control owner SID");
  std::vector<unsigned char> data(bytes);
  if (!GetTokenInformation(token, TokenUser, data.data(), bytes, &bytes))
    throw ControlError(16, "cannot query control owner SID");
  const auto* user = reinterpret_cast<const TOKEN_USER*>(data.data());
  const DWORD sid_bytes = GetLengthSid(user->User.Sid);
  std::vector<unsigned char> sid(sid_bytes);
  if (!CopySid(sid_bytes, sid.data(), user->User.Sid))
    throw ControlError(16, "cannot copy control owner SID");
  return sid;
}

std::vector<unsigned char> CurrentSid() {
  HANDLE raw = nullptr;
  if (!OpenProcessToken(GetCurrentProcess(), TOKEN_QUERY, &raw))
    throw ControlError(16, "cannot open control owner token");
  Handle token(raw);
  return TokenSid(token.Get());
}

std::wstring SidText(const std::vector<unsigned char>& sid) {
  wchar_t* text = nullptr;
  if (!ConvertSidToStringSidW(const_cast<unsigned char*>(sid.data()), &text))
    throw ControlError(16, "cannot encode control owner SID");
  const std::wstring result(text);
  LocalFree(text);
  return result;
}

class OwnerSecurity {
public:
  explicit OwnerSecurity(bool inherit = false) : sid_(CurrentSid()) {
    const auto sid = SidText(sid_);
    const auto sddl = L"O:" + sid + L"D:P(A;" +
                      (inherit ? std::wstring(L"OICI") : std::wstring()) +
                      L";GA;;;" + sid + L")";
    if (!ConvertStringSecurityDescriptorToSecurityDescriptorW(
            sddl.c_str(), SDDL_REVISION_1, &descriptor_, nullptr))
      throw ControlError(16, "cannot set private control security");
    attributes_.nLength = sizeof(attributes_);
    attributes_.lpSecurityDescriptor = descriptor_;
    attributes_.bInheritHandle = FALSE;
  }
  ~OwnerSecurity() { LocalFree(descriptor_); }
  OwnerSecurity(const OwnerSecurity&) = delete;
  OwnerSecurity& operator=(const OwnerSecurity&) = delete;
  SECURITY_ATTRIBUTES* Attributes() { return &attributes_; }
  PSID Sid() { return sid_.data(); }
  PACL Dacl() const {
    BOOL present = FALSE;
    BOOL defaulted = FALSE;
    PACL acl = nullptr;
    if (!GetSecurityDescriptorDacl(descriptor_, &present, &acl, &defaulted) ||
        !present || !acl)
      throw ControlError(16, "cannot read private control security");
    return acl;
  }

private:
  std::vector<unsigned char> sid_;
  PSECURITY_DESCRIPTOR descriptor_ = nullptr;
  SECURITY_ATTRIBUTES attributes_{};
};

bool PrivateOwner(HANDLE handle, SE_OBJECT_TYPE type = SE_FILE_OBJECT) {
  const auto sid = CurrentSid();
  PSID owner = nullptr;
  PACL acl = nullptr;
  PSECURITY_DESCRIPTOR descriptor = nullptr;
  if (GetSecurityInfo(
          handle, type, OWNER_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION,
          &owner, nullptr, &acl, nullptr, &descriptor) != ERROR_SUCCESS)
    return false;
  bool valid = owner &&
               EqualSid(owner, const_cast<unsigned char*>(sid.data())) && acl &&
               acl->AceCount > 0;
  SECURITY_DESCRIPTOR_CONTROL control = 0;
  DWORD revision = 0;
  valid = valid &&
          GetSecurityDescriptorControl(descriptor, &control, &revision) &&
          (control & SE_DACL_PROTECTED) != 0;
  // Windows can split a generic inheritable directory ACE into an effective
  // ACE and an inherit-only ACE. Every grant must still name only this SID.
  for (DWORD index = 0; valid && index < acl->AceCount; ++index) {
    void* raw = nullptr;
    if (!GetAce(acl, index, &raw)) {
      valid = false;
      break;
    }
    const auto* ace = static_cast<const ACCESS_ALLOWED_ACE*>(raw);
    valid = ace->Header.AceType == ACCESS_ALLOWED_ACE_TYPE &&
            EqualSid(const_cast<DWORD*>(&ace->SidStart),
                     const_cast<unsigned char*>(sid.data()));
  }
  LocalFree(descriptor);
  return valid;
}

void RejectReparse(HANDLE handle) {
  FILE_ATTRIBUTE_TAG_INFO info{};
  if (!GetFileInformationByHandleEx(handle, FileAttributeTagInfo, &info,
                                    sizeof(info)) ||
      (info.FileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0)
    throw ControlError(16, "control storage cannot be a reparse point");
}

void EnsureDirectory(const fs::path& path, bool protect = true) {
  OwnerSecurity security(true);
  if (!CreateDirectoryW(path.c_str(), security.Attributes()) &&
      GetLastError() != ERROR_ALREADY_EXISTS)
    throw ControlError(16, "cannot create private discovery directory");
  Handle directory(CreateFileW(
      path.c_str(), READ_CONTROL | WRITE_DAC,
      FILE_SHARE_READ | FILE_SHARE_WRITE, nullptr, OPEN_EXISTING,
      FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT, nullptr));
  if (!directory)
    throw ControlError(16, "cannot open private discovery directory");
  RejectReparse(directory.Get());
  PSID owner = nullptr;
  PSECURITY_DESCRIPTOR descriptor = nullptr;
  const auto result = GetSecurityInfo(directory.Get(), SE_FILE_OBJECT,
                                      OWNER_SECURITY_INFORMATION, &owner,
                                      nullptr, nullptr, nullptr, &descriptor);
  const bool owned =
      result == ERROR_SUCCESS && owner && EqualSid(owner, security.Sid());
  LocalFree(descriptor);
  if (!owned)
    throw ControlError(16, "discovery directory has a different owner");
  if (!protect)
    return;
  if (SetSecurityInfo(
          directory.Get(), SE_FILE_OBJECT,
          DACL_SECURITY_INFORMATION | PROTECTED_DACL_SECURITY_INFORMATION,
          nullptr, nullptr, security.Dacl(), nullptr) != ERROR_SUCCESS ||
      !PrivateOwner(directory.Get()))
    throw ControlError(16, "cannot protect discovery directory");
}

std::string CreationTime(HANDLE process) {
  FILETIME creation{}, exit{}, kernel{}, user{};
  if (!GetProcessTimes(process, &creation, &exit, &kernel, &user))
    throw ControlError(16, "cannot query process creation time");
  const auto ticks =
      (static_cast<std::uint64_t>(creation.dwHighDateTime) << 32) |
      creation.dwLowDateTime;
  std::array<char, 21> buffer{};
  std::snprintf(buffer.data(), buffer.size(), "%llu",
                static_cast<unsigned long long>(ticks));
  return buffer.data();
}

std::wstring CanonicalPath(std::wstring_view path) {
  Handle file(
      CreateFileW(std::wstring(path).c_str(), FILE_READ_ATTRIBUTES,
                  FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                  nullptr, OPEN_EXISTING, 0, nullptr));
  if (!file)
    throw ControlError(16, "cannot resolve canonical executable path");
  std::wstring result(32768, L'\0');
  const DWORD count = GetFinalPathNameByHandleW(
      file.Get(), result.data(), static_cast<DWORD>(result.size()),
      FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
  if (count == 0 || count >= result.size())
    throw ControlError(16, "cannot resolve canonical executable path");
  result.resize(count);
  if (result.starts_with(L"\\\\?\\UNC\\"))
    result = L"\\\\" + result.substr(8);
  else if (result.starts_with(L"\\\\?\\"))
    result.erase(0, 4);
  return result;
}

std::string ProcessPath(HANDLE process) {
  std::wstring result(32768, L'\0');
  DWORD count = static_cast<DWORD>(result.size());
  if (!QueryFullProcessImageNameW(process, 0, result.data(), &count))
    throw ControlError(16, "cannot query executable identity");
  result.resize(count);
  return Utf8(CanonicalPath(result));
}

bool EqualPath(std::string_view lhs, std::string_view rhs) {
  const auto left = Wide(lhs);
  const auto right = Wide(rhs);
  return CompareStringOrdinal(left.data(), static_cast<int>(left.size()),
                              right.data(), static_cast<int>(right.size()),
                              TRUE) == CSTR_EQUAL;
}

bool EqualIdentity(const InstanceIdentity& lhs, const InstanceIdentity& rhs) {
  return lhs.instance_id == rhs.instance_id && lhs.pid == rhs.pid &&
         lhs.process_creation_time == rhs.process_creation_time &&
         EqualPath(lhs.exe_path, rhs.exe_path);
}

void VerifyProcess(HANDLE process, const InstanceIdentity& identity) {
  if (WaitForSingleObject(process, 0) != WAIT_TIMEOUT ||
      CreationTime(process) != identity.process_creation_time ||
      !EqualPath(ProcessPath(process), identity.exe_path) ||
      !EqualPath(Utf8(CanonicalPath(Wide(identity.exe_path))),
                 identity.exe_path))
    throw ControlError(16, "process identity does not match discovery");
  HANDLE raw = nullptr;
  if (!OpenProcessToken(process, TOKEN_QUERY, &raw))
    throw ControlError(16, "cannot query instance owner");
  Handle token(raw);
  const auto owner = TokenSid(token.Get());
  const auto current = CurrentSid();
  if (!EqualSid(const_cast<unsigned char*>(owner.data()),
                const_cast<unsigned char*>(current.data())))
    throw ControlError(16, "instance has a different owner");
}

DWORD Remaining(ULONGLONG deadline) {
  const auto now = GetTickCount64();
  if (now >= deadline)
    return 0;
  return static_cast<DWORD>(std::min<ULONGLONG>(deadline - now, MAXDWORD - 1));
}

struct PendingIo {
  Handle pipe;
  Handle event;
  OVERLAPPED operation{};
  std::vector<unsigned char> bytes;
};

void CancelPending(std::unique_ptr<PendingIo> pending) {
  CancelIoEx(pending->pipe.Get(), &pending->operation);
  if (WaitForSingleObject(pending->event.Get(), 0) == WAIT_OBJECT_0)
    return;
  // Cancellation completion is asynchronous. Return at the deadline while
  // keeping OVERLAPPED and private bytes alive until the kernel completes.
  // This worker owns a duplicate handle; it has no server/inference state.
  auto* owned = pending.release();
  try {
    std::thread([owned] {
      WaitForSingleObject(owned->event.Get(), INFINITE);
      delete owned;
    }).detach();
  } catch (const std::exception&) {
    // If thread creation fails, process exit reclaims this allocation and
    // its handles. Never release storage still referenced by kernel I/O.
  }
}

bool Transfer(HANDLE pipe, bool write, void* data, DWORD size, DWORD& count,
              ULONGLONG deadline, HANDLE stop = nullptr) {
  auto pending = std::make_unique<PendingIo>();
  pending->event.Reset(CreateEventW(nullptr, TRUE, FALSE, nullptr));
  HANDLE duplicate = nullptr;
  if (!pending->event ||
      !DuplicateHandle(GetCurrentProcess(), pipe, GetCurrentProcess(),
                       &duplicate, 0, FALSE, DUPLICATE_SAME_ACCESS))
    return false;
  pending->pipe.Reset(duplicate);
  pending->operation.hEvent = pending->event.Get();
  pending->bytes.resize(size);
  if (write)
    std::copy_n(static_cast<const unsigned char*>(data), size,
                pending->bytes.data());
  const BOOL immediate =
      write ? WriteFile(pending->pipe.Get(), pending->bytes.data(), size,
                        &count, &pending->operation)
            : ReadFile(pending->pipe.Get(), pending->bytes.data(), size, &count,
                       &pending->operation);
  if (immediate) {
    if (!write)
      std::copy_n(pending->bytes.data(), count,
                  static_cast<unsigned char*>(data));
    return true;
  }
  if (GetLastError() != ERROR_IO_PENDING)
    return false;
  const HANDLE handles[] = {pending->event.Get(), stop};
  const DWORD result =
      WaitForMultipleObjects(stop ? 2 : 1, handles, FALSE, Remaining(deadline));
  if (result != WAIT_OBJECT_0) {
    CancelPending(std::move(pending));
    return false;
  }
  if (!GetOverlappedResult(pending->pipe.Get(), &pending->operation, &count,
                           FALSE))
    return false;
  if (!write)
    std::copy_n(pending->bytes.data(), count,
                static_cast<unsigned char*>(data));
  return true;
}

bool WriteLine(HANDLE pipe, const Value& value, ULONGLONG deadline,
               HANDLE stop = nullptr) {
  auto line = value.dump();
  if (line.size() >= kRecordBytes)
    return false;
  line += '\n';
  std::size_t offset = 0;
  while (offset < line.size()) {
    DWORD count = 0;
    if (!Transfer(pipe, true, line.data() + offset,
                  static_cast<DWORD>(line.size() - offset), count, deadline,
                  stop) ||
        count == 0)
      return false;
    offset += count;
  }
  return true;
}

bool ReadLine(HANDLE pipe, Value& value, std::string& buffered,
              std::size_t limit, ULONGLONG deadline, HANDLE stop = nullptr) {
  for (;;) {
    const auto newline = buffered.find('\n');
    if (newline != std::string::npos) {
      if (newline > limit)
        return false;
      try {
        value = metadata::Parse(std::string_view(buffered.data(), newline));
        buffered.erase(0, newline + 1);
        return value.is_object();
      } catch (const std::exception&) {
        return false;
      }
    }
    if (buffered.size() > limit || Remaining(deadline) == 0)
      return false;
    std::array<char, 4096> data{};
    DWORD count = 0;
    if (!Transfer(pipe, false, data.data(),
                  static_cast<DWORD>(std::min<std::size_t>(
                      data.size(), limit + 1 - buffered.size())),
                  count, deadline, stop) ||
        count == 0)
      return false;
    buffered.append(data.data(), count);
  }
}

bool OwnerClient(HANDLE pipe) {
  if (!ImpersonateNamedPipeClient(pipe))
    return false;
  HANDLE raw = nullptr;
  const bool opened =
      OpenThreadToken(GetCurrentThread(), TOKEN_QUERY, TRUE, &raw) != 0;
  // Revert before querying our process SID or dispatching server callbacks.
  const bool reverted = RevertToSelf() != 0;
  Handle token(raw);
  if (!opened || !reverted)
    return false;
  try {
    const auto sid = TokenSid(token.Get());
    const auto owner = CurrentSid();
    return EqualSid(const_cast<unsigned char*>(sid.data()),
                    const_cast<unsigned char*>(owner.data())) != 0;
  } catch (const std::exception&) {
    return false;
  }
}

Handle OpenPrivateRecord(const fs::path& path) {
  Handle file(CreateFileW(path.c_str(), GENERIC_READ | READ_CONTROL,
                          FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr,
                          OPEN_EXISTING, FILE_FLAG_OPEN_REPARSE_POINT,
                          nullptr));
  if (!file)
    throw ControlError(16, "cannot open discovery record");
  RejectReparse(file.Get());
  if (!PrivateOwner(file.Get()))
    throw ControlError(16, "discovery record is not private to its owner");
  return file;
}

std::string ReadRecord(const fs::path& path) {
  auto file = OpenPrivateRecord(path);
  LARGE_INTEGER length{};
  if (!GetFileSizeEx(file.Get(), &length) || length.QuadPart < 0 ||
      length.QuadPart > static_cast<LONGLONG>(kRecordBytes))
    throw ControlError(16, "discovery record exceeds its size limit");
  std::string data(static_cast<std::size_t>(length.QuadPart), '\0');
  DWORD read = 0;
  if (!ReadFile(file.Get(), data.data(), static_cast<DWORD>(data.size()), &read,
                nullptr) ||
      read != data.size())
    throw ControlError(16, "cannot read discovery record");
  return data;
}

std::string NormalHost(std::string host) {
  if (host == "localhost")
    host = "127.0.0.1";
  std::array<char, INET6_ADDRSTRLEN> output{};
  in_addr ipv4{};
  in6_addr ipv6{};
  if (InetPtonA(AF_INET, host.c_str(), &ipv4) == 1 &&
      InetNtopA(AF_INET, &ipv4, output.data(), output.size()))
    return output.data();
  if (InetPtonA(AF_INET6, host.c_str(), &ipv6) == 1 &&
      InetNtopA(AF_INET6, &ipv6, output.data(), output.size()))
    return output.data();
  throw ControlError(2, "serve host must be an IP address or localhost");
}
#endif
}  // namespace

ControlError::ControlError(int exit_code, const std::string& message)
    : std::runtime_error(message), exit_code_(exit_code) {}

ControlError::ControlError(int exit_code, const std::string& message, Value details)
    : std::runtime_error(message), exit_code_(exit_code), details_(std::move(details)) {}

bool ValidInstanceId(std::string_view instance_id) {
  return Hex(instance_id, 32);
}

std::string ControlPipeName(std::string_view instance_id) {
  if (!ValidInstanceId(instance_id))
    throw ControlError(16, "invalid instance ID");
  return "\\\\.\\pipe\\Bubo.Control." + std::string(instance_id);
}

InstanceIdentity CurrentIdentity(std::string instance_id) {
  if (!ValidInstanceId(instance_id))
    throw ControlError(16, "invalid instance ID");
#ifdef _WIN32
  return {std::move(instance_id), GetCurrentProcessId(),
          CreationTime(GetCurrentProcess()), ProcessPath(GetCurrentProcess())};
#else
  Unsupported();
#endif
}

Value IdentityJson(const InstanceIdentity& identity) {
  auto value = Value::object();
  value["instance_id"] = identity.instance_id;
  value["pid"] = static_cast<unsigned long long>(identity.pid);
  value["process_creation_time"] = identity.process_creation_time;
  value["exe_path"] = identity.exe_path;
  return value;
}

fs::path DiscoveryDirectory(std::string_view root) {
  const auto selected =
      root.empty() ? EnvironmentUtf8("LOCALAPPDATA") : std::string(root);
  if (selected.empty())
    throw ControlError(16, "LOCALAPPDATA is not available");
  const auto path = fs::u8path(selected);
  if (!path.is_absolute())
    throw ControlError(16, "LOCALAPPDATA must be an absolute path");
  return path / "Bubo" / "instances";
}

Value DiscoveryJson(const DiscoveryRecord& record) {
  auto value = Value::object();
  value["schema_version"] = 1;
  value["control_version"] = kControlVersion;
  value["identity"] = IdentityJson(record.identity);
  value["endpoint"] =
      record.endpoint.empty() ? Value() : Value(record.endpoint);
  value["phase"] = record.phase;
  value["log_path"] = record.log_path ? Value(*record.log_path) : Value();
  // A discovery record deliberately has no snapshot or effective settings:
  // those can contain secrets and must be obtained through verified control.
  value["last_error"] = RedactedError(record.last_error);
  return value;
}

void WriteDiscovery(const DiscoveryRecord& record, std::string_view root) {
  (void)ParseIdentity(IdentityJson(record.identity));
#ifdef _WIN32
  if (!EqualIdentity(CurrentIdentity(record.identity.instance_id),
                     record.identity))
    throw ControlError(16,
                       "only the identified process can publish its record");
  const auto directory = DiscoveryDirectory(root);
  EnsureDirectory(directory.parent_path(), false);
  EnsureDirectory(directory);
  const auto path = directory / (record.identity.instance_id + ".json");
  const auto temporary =
      directory / (record.identity.instance_id + "." +
                   std::to_string(GetCurrentThreadId()) + ".tmp");
  OwnerSecurity security;
  Handle file(CreateFileW(
      temporary.c_str(), GENERIC_WRITE | READ_CONTROL, 0, security.Attributes(),
      CREATE_NEW, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_WRITE_THROUGH, nullptr));
  if (!file)
    throw ControlError(16, "cannot create private discovery record");
  try {
    auto text = DiscoveryJson(record).dump();
    text += '\n';
    if (text.size() > kRecordBytes)
      throw ControlError(16, "discovery record exceeds its size limit");
    DWORD written = 0;
    if (!WriteFile(file.Get(), text.data(), static_cast<DWORD>(text.size()),
                   &written, nullptr) ||
        written != text.size() || !FlushFileBuffers(file.Get()) ||
        !PrivateOwner(file.Get()))
      throw ControlError(16, "cannot write private discovery record");
    file.Reset();
    const auto attributes = GetFileAttributesW(path.c_str());
    if (attributes != INVALID_FILE_ATTRIBUTES &&
        (attributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0)
      throw ControlError(16, "discovery record cannot be a reparse point");
    // Windows can deny replacement while even a share-delete reader is open.
    // Prepare's rename helper has a 10-second payload deadline and excludes
    // access-denied; control publication needs a short, bounded retry while
    // preserving its owner-only temporary file and typed failure.
    const auto publish_deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
    while (!MoveFileExW(temporary.c_str(), path.c_str(),
                        MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
      const auto code = GetLastError();
      if ((code != ERROR_ACCESS_DENIED && code != ERROR_SHARING_VIOLATION &&
           code != ERROR_LOCK_VIOLATION) ||
          std::chrono::steady_clock::now() >= publish_deadline)
        throw ControlError(16, "cannot publish atomic discovery record");
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  } catch (...) {
    file.Reset();
    // Only this invocation's CREATE_NEW temporary file is removed.
    DeleteFileW(temporary.c_str());
    throw;
  }
#else
  (void)root;
  Unsupported();
#endif
}

void WriteTombstone(const DiscoveryRecord& record, const Value& last_error,
                    std::string_view root) {
  auto tombstone = record;
  tombstone.phase = "stopped";
  tombstone.last_error = RedactedError(last_error);
  WriteDiscovery(tombstone, root);
}

std::optional<DiscoveryRecord> ReadDiscovery(std::string_view instance_id,
                                             std::string_view root) {
  if (!ValidInstanceId(instance_id))
    throw ControlError(16, "invalid instance ID");
#ifdef _WIN32
  const auto path =
      DiscoveryDirectory(root) / (std::string(instance_id) + ".json");
  if (GetFileAttributesW(path.c_str()) == INVALID_FILE_ATTRIBUTES) {
    const auto error = GetLastError();
    if (error == ERROR_FILE_NOT_FOUND || error == ERROR_PATH_NOT_FOUND)
      return std::nullopt;
  }
  try {
    const auto document = metadata::Parse(ReadRecord(path));
    metadata::Keys(document, {"schema_version", "control_version", "identity",
                              "endpoint", "phase", "log_path", "last_error"});
    CheckVersions(document);
    DiscoveryRecord record;
    record.identity = ParseIdentity(metadata::Required(document, "identity"));
    if (record.identity.instance_id != instance_id)
      throw ControlError(16, "discovery filename identity does not match");
    const auto& endpoint = metadata::Required(document, "endpoint");
    if (!endpoint.is_null())
      record.endpoint = metadata::String(endpoint);
    record.phase = metadata::String(metadata::Required(document, "phase"));
    if (record.phase != "starting" && record.phase != "loading" &&
        record.phase != "ready" && record.phase != "stopping" &&
        record.phase != "stopped" && record.phase != "error")
      throw ControlError(16, "invalid discovery phase");
    const auto& log = metadata::Required(document, "log_path");
    if (!log.is_null())
      record.log_path = metadata::String(log);
    record.last_error =
        RedactedError(metadata::Required(document, "last_error"));
    return record;
  } catch (const ControlError&) {
    throw;
  } catch (const std::exception&) {
    throw ControlError(16, "invalid discovery record");
  }
#else
  (void)root;
  Unsupported();
#endif
}

std::vector<DiscoveryRecord> DiscoverRecords(std::string_view root) {
  std::vector<DiscoveryRecord> records;
  std::error_code error;
  const auto directory = DiscoveryDirectory(root);
  if (!fs::exists(directory, error)) {
    if (error)
      throw ControlError(16, "cannot inspect discovery directory");
    return records;
  }
#ifdef _WIN32
  Handle handle(CreateFileW(
      directory.c_str(), READ_CONTROL, FILE_SHARE_READ | FILE_SHARE_WRITE,
      nullptr, OPEN_EXISTING,
      FILE_FLAG_BACKUP_SEMANTICS | FILE_FLAG_OPEN_REPARSE_POINT, nullptr));
  if (!handle)
    throw ControlError(16, "cannot inspect discovery directory");
  RejectReparse(handle.Get());
  if (!PrivateOwner(handle.Get()))
    throw ControlError(16, "discovery directory is not private to its owner");
  for (const auto& entry : fs::directory_iterator(directory)) {
    const auto filename = entry.path().filename().u8string();
    const std::string name(filename.begin(), filename.end());
    if (name.size() != 37 || !name.ends_with(".json") ||
        !ValidInstanceId(std::string_view(name).substr(0, 32)))
      continue;
    try {
      auto record = ReadDiscovery(std::string_view(name).substr(0, 32), root);
      if (record)
        records.push_back(std::move(*record));
    } catch (const ControlError&) {
      // Discovery is a hint. An invalid entry cannot become an instance.
    }
  }
  return records;
#else
  // Portable prepared status has no native Windows instances to discover.
  // Direct identity and control actions remain explicitly unsupported.
  return records;
#endif
}

bool RecordedStopped(const DiscoveryRecord& record) {
#ifdef _WIN32
  if (record.phase != "stopped" ||
      !ValidInstanceId(record.identity.instance_id))
    return false;
  try {
    (void)ParseIdentity(IdentityJson(record.identity));
    if (!EqualPath(Utf8(CanonicalPath(Wide(record.identity.exe_path))),
                   record.identity.exe_path))
      return false;
  } catch (const std::exception&) {
    return false;
  }
  // The caller obtained this record through ReadDiscovery, which checked the
  // owner and protected ACL. A tombstone alone cannot authorize live control.
  Handle process(OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION,
                             FALSE, record.identity.pid));
  if (!process)
    return GetLastError() == ERROR_INVALID_PARAMETER;
  if (CreationTime(process.Get()) != record.identity.process_creation_time)
    return false;
  // A retained, exited process object can no longer answer an image-path
  // query. Its matching creation time and signalled handle confirm exit;
  // the record's canonical path and owner security were checked above.
  return WaitForSingleObject(process.Get(), 0) == WAIT_OBJECT_0;
#else
  (void)record;
  Unsupported();
#endif
}

class ControlServer::Impl {
public:
  Impl(InstanceIdentity id, std::function<Value()> sample,
       std::function<void()> request_stop)
      : identity(std::move(id)),
        snapshot(std::move(sample)),
        stop(std::move(request_stop)) {}
  InstanceIdentity identity;
  std::function<Value()> snapshot;
  std::function<void()> stop;
#ifdef _WIN32
  Handle stopping;
  Handle pipe;
  std::thread thread;

  Handle NewPipe(bool first) {
    OwnerSecurity security;
    const auto name = Wide(ControlPipeName(identity.instance_id));
    Handle handle(CreateNamedPipeW(
        name.c_str(),
        PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED |
            (first ? FILE_FLAG_FIRST_PIPE_INSTANCE : 0),
        PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT |
            PIPE_REJECT_REMOTE_CLIENTS,
        1, 65536, 65536, kControlDeadlineMs, security.Attributes()));
    if (!handle || !PrivateOwner(handle.Get()))
      throw ControlError(16, "cannot create private local control pipe");
    return handle;
  }

  bool Connect() {
    auto pending = std::make_unique<PendingIo>();
    pending->event.Reset(CreateEventW(nullptr, TRUE, FALSE, nullptr));
    HANDLE duplicate = nullptr;
    if (!pending->event ||
        !DuplicateHandle(GetCurrentProcess(), pipe.Get(), GetCurrentProcess(),
                         &duplicate, 0, FALSE, DUPLICATE_SAME_ACCESS))
      return false;
    pending->pipe.Reset(duplicate);
    pending->operation.hEvent = pending->event.Get();
    if (ConnectNamedPipe(pending->pipe.Get(), &pending->operation))
      return true;
    const DWORD error = GetLastError();
    if (error == ERROR_PIPE_CONNECTED)
      return true;
    if (error != ERROR_IO_PENDING)
      return false;
    const HANDLE handles[] = {stopping.Get(), pending->event.Get()};
    const auto wait = WaitForMultipleObjects(2, handles, FALSE, INFINITE);
    DWORD count = 0;
    if (wait != WAIT_OBJECT_0 + 1) {
      CancelPending(std::move(pending));
      return false;
    }
    return GetOverlappedResult(pending->pipe.Get(), &pending->operation, &count,
                               FALSE) != 0;
  }

  void Receipt(std::string& buffered, ULONGLONG deadline) {
    Value receipt;
    if (!ReadLine(pipe.Get(), receipt, buffered, kRequestBytes, deadline,
                  stopping.Get()))
      return;
    metadata::Keys(receipt,
                   {"schema_version", "control_version", "action", "identity"});
    CheckVersions(receipt);
    if (receipt.member_str("action") != "received" ||
        !EqualIdentity(ParseIdentity(metadata::Required(receipt, "identity")),
                       identity))
      throw ControlError(16, "invalid local control response receipt");
  }

  void Client() {
    bool handshaken = false;
    std::string buffered;
    // A connection has a bounded lifetime; abandoned clients cannot keep the
    // single pipe instance occupied indefinitely or stall server shutdown.
    const auto deadline = GetTickCount64() + kControlDeadlineMs;
    for (unsigned int requests = 0; requests < 3; ++requests) {
      Value request;
      if (!ReadLine(pipe.Get(), request, buffered, kRequestBytes, deadline,
                    stopping.Get()) ||
          !OwnerClient(pipe.Get()))
        return;
      try {
        metadata::Keys(request, {"schema_version", "control_version", "action",
                                 "identity"});
        CheckVersions(request);
        const auto claimed =
            ParseIdentity(metadata::Required(request, "identity"));
        if (!EqualIdentity(claimed, identity))
          throw ControlError(16, "control identity does not match");
        const auto action =
            metadata::String(metadata::Required(request, "action"));
        if (action == "handshake" && !handshaken) {
          auto response = Protocol("handshake", identity);
          response["accepted"] = true;
          if (!WriteLine(pipe.Get(), response, deadline, stopping.Get()))
            return;
          handshaken = true;
        } else if (action == "status" && handshaken) {
          auto response = Protocol("status", identity);
          response["snapshot"] = snapshot();
          if (WriteLine(pipe.Get(), response, deadline, stopping.Get()))
            Receipt(buffered, deadline);
          return;
        } else if (action == "stop" && handshaken) {
          auto response = Protocol("stop", identity);
          response["acknowledged"] = true;
          response["stopped"] = false;
          // Acknowledgement means accepted only. The client confirms exit on
          // its retained process handle; no pipe message promises exit.
          if (WriteLine(pipe.Get(), response, deadline, stopping.Get())) {
            // DisconnectNamedPipe discards unread buffered responses. An
            // explicit receipt makes acknowledgement observable without a
            // blocking FlushFileBuffers call. Stop still proceeds on loss.
            try {
              Receipt(buffered, deadline);
            } catch (const std::exception&) {
            }
          }
          stop();
          return;
        } else {
          throw ControlError(16, "local control handshake required");
        }
      } catch (const ControlError& error) {
        auto response = Protocol("error", identity);
        response["exit_code"] = error.ExitCode();
        response["code"] =
            error.ExitCode() == 15 ? "protocol_mismatch" : "control_access";
        if (WriteLine(pipe.Get(), response, deadline, stopping.Get()))
          Receipt(buffered, deadline);
        return;
      } catch (const std::exception&) {
        auto response = Protocol("error", identity);
        response["exit_code"] = 16;
        response["code"] = "control_access";
        if (WriteLine(pipe.Get(), response, deadline, stopping.Get()))
          Receipt(buffered, deadline);
        return;
      }
    }
  }

  void Run() noexcept {
    while (WaitForSingleObject(stopping.Get(), 0) == WAIT_TIMEOUT) {
      if (!Connect())
        break;
      try {
        Client();
      } catch (...) {
        // An invalid/failed client cannot bring down inference.
      }
      DisconnectNamedPipe(pipe.Get());
    }
  }
#endif
};

ControlServer::ControlServer(InstanceIdentity identity,
                             std::function<Value()> snapshot,
                             std::function<void()> stop)
    : impl_(std::make_unique<Impl>(std::move(identity), std::move(snapshot),
                                   std::move(stop))) {}
ControlServer::~ControlServer() {
  Shutdown();
}
void ControlServer::Start() {
#ifdef _WIN32
  if (impl_->thread.joinable())
    throw ControlError(16, "local control server is already running");
  if (!EqualIdentity(CurrentIdentity(impl_->identity.instance_id),
                     impl_->identity))
    throw ControlError(16, "control server has a different process identity");
  impl_->stopping.Reset(CreateEventW(nullptr, TRUE, FALSE, nullptr));
  if (!impl_->stopping)
    throw ControlError(16, "cannot create local control cancellation event");
  impl_->pipe = impl_->NewPipe(true);
  impl_->thread = std::thread([server = impl_.get()] { server->Run(); });
#else
  Unsupported();
#endif
}
void ControlServer::Shutdown() {
#ifdef _WIN32
  if (impl_->stopping)
    SetEvent(impl_->stopping.Get());
  if (impl_->thread.joinable())
    impl_->thread.join();
  impl_->pipe.Reset();
  impl_->stopping.Reset();
#endif
}

class VerifiedInstance::Impl {
public:
  InstanceIdentity identity;
  std::uint32_t timeout_ms = kControlDeadlineMs;
#ifdef _WIN32
  Handle process;
  // A fresh handshake precedes each request. No idle connection can starve
  // other clients, and identity is rechecked immediately before Stop.
  Value Request(std::string_view action) {
    VerifyProcess(process.Get(), identity);
    const auto name = Wide(ControlPipeName(identity.instance_id));
    const auto deadline = GetTickCount64() + timeout_ms;
    Handle pipe;
    do {
      pipe.Reset(CreateFileW(name.c_str(),
                             GENERIC_READ | GENERIC_WRITE | READ_CONTROL, 0,
                             nullptr, OPEN_EXISTING,
                             FILE_FLAG_OVERLAPPED | SECURITY_SQOS_PRESENT |
                                 SECURITY_IDENTIFICATION,
                             nullptr));
      if (pipe)
        break;
      const auto error = GetLastError();
      if (error != ERROR_PIPE_BUSY && error != ERROR_FILE_NOT_FOUND)
        throw ControlError(16, "cannot access local control pipe");
      const DWORD remaining = Remaining(deadline);
      if (remaining == 0)
        throw ControlError(16, "local control connection deadline expired");
      // WaitNamedPipe does not wait for a missing pipe. A bounded event wait
      // avoids a spin while the process is creating/replacing its listener.
      if (!WaitNamedPipeW(name.c_str(), std::min<DWORD>(remaining, 50))) {
        const auto wait_error = GetLastError();
        if (wait_error == ERROR_FILE_NOT_FOUND) {
          if (WaitForSingleObject(process.Get(),
                                  std::min<DWORD>(remaining, 10)) !=
              WAIT_TIMEOUT)
            throw ControlError(16, "instance exited before control handshake");
        } else if (wait_error != ERROR_SEM_TIMEOUT &&
                   wait_error != ERROR_PIPE_BUSY) {
          throw ControlError(16, "cannot wait for local control pipe");
        }
      }
    } while (Remaining(deadline));
    if (!pipe || !PrivateOwner(pipe.Get()))
      throw ControlError(16, "local control pipe is not private to its owner");
    ULONG server_pid = 0;
    if (!GetNamedPipeServerProcessId(pipe.Get(), &server_pid) ||
        server_pid != identity.pid)
      throw ControlError(16, "control pipe belongs to a different process");
    VerifyProcess(process.Get(), identity);
    std::string buffered;
    const auto exchange = [&](std::string_view requested) {
      if (!WriteLine(pipe.Get(), Protocol(requested, identity), deadline))
        throw ControlError(16, "local control request failed");
      Value response;
      if (!ReadLine(pipe.Get(), response, buffered, kRecordBytes - 1, deadline))
        throw ControlError(16, "local control response failed");
      CheckVersions(response);
      const auto returned =
          ParseIdentity(metadata::Required(response, "identity"));
      if (!EqualIdentity(returned, identity))
        throw ControlError(16, "local control handshake identity failed");
      const auto returned_action = response.member_str("action");
      if (returned_action == "error") {
        const auto code =
            metadata::Integer(metadata::Required(response, "exit_code"));
        throw ControlError(code == 15 ? 15 : 16,
                           "local control request rejected");
      }
      if (returned_action != requested)
        throw ControlError(16, "local control action does not match");
      return response;
    };
    auto handshake = exchange("handshake");
    const auto* accepted = handshake.find("accepted");
    if (!accepted || !accepted->is_bool() || !accepted->as_bool())
      throw ControlError(16, "local control handshake rejected");
    VerifyProcess(process.Get(), identity);
    if (action == "handshake")
      return handshake;
    auto response = exchange(action);
    // The response was consumed before this receipt. It is safe for the
    // server to disconnect, and Stop can enter its existing shutdown path.
    WriteLine(pipe.Get(), Protocol("received", identity), deadline);
    return response;
  }
#endif
};

VerifiedInstance::VerifiedInstance(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
VerifiedInstance::~VerifiedInstance() = default;
VerifiedInstance::VerifiedInstance(VerifiedInstance&&) noexcept = default;
VerifiedInstance& VerifiedInstance::operator=(VerifiedInstance&&) noexcept =
    default;
const InstanceIdentity& VerifiedInstance::Identity() const {
  return impl_->identity;
}
Value VerifiedInstance::Snapshot() {
#ifdef _WIN32
  try {
    const auto response = impl_->Request("status");
    const auto& snapshot = metadata::Required(response, "snapshot");
    if (!snapshot.is_object())
      throw ControlError(16, "local control status is not an object");
    return snapshot;
  } catch (const ControlError&) {
    throw;
  } catch (const std::exception&) {
    throw ControlError(16, "invalid local control status response");
  }
#else
  Unsupported();
#endif
}
Value VerifiedInstance::RequestStop() {
#ifdef _WIN32
  try {
    const auto response = impl_->Request("stop");
    const auto* ack = response.find("acknowledged");
    const auto* stopped = response.find("stopped");
    if (!ack || !ack->is_bool() || !ack->as_bool() || !stopped ||
        !stopped->is_bool() || stopped->as_bool())
      throw ControlError(16, "invalid stop acknowledgement");
    return response;
  } catch (const ControlError&) {
    throw;
  } catch (const std::exception&) {
    throw ControlError(16, "invalid local control stop response");
  }
#else
  Unsupported();
#endif
}
bool VerifiedInstance::WaitForExit(std::uint32_t wait_ms) {
#ifdef _WIN32
  const auto result = WaitForSingleObject(
      impl_->process.Get(), std::min<DWORD>(wait_ms, MAXDWORD - 1));
  if (result != WAIT_OBJECT_0 && result != WAIT_TIMEOUT)
    throw ControlError(16, "cannot wait for identified process exit");
  return result == WAIT_OBJECT_0;
#else
  (void)wait_ms;
  Unsupported();
#endif
}
VerifiedInstance VerifiedConnect(const DiscoveryRecord& record,
                                 std::uint32_t timeout_ms) {
#ifdef _WIN32
  try {
    auto impl = std::make_unique<VerifiedInstance::Impl>();
    impl->identity = ParseIdentity(IdentityJson(record.identity));
    impl->timeout_ms = std::min<std::uint32_t>(timeout_ms, kControlDeadlineMs);
    impl->process.Reset(
        OpenProcess(SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION, FALSE,
                    record.identity.pid));
    if (!impl->process)
      throw ControlError(16, "cannot open identified process");
    impl->Request("handshake");
    return VerifiedInstance(std::move(impl));
  } catch (const ControlError&) {
    throw;
  } catch (const std::exception&) {
    throw ControlError(16, "invalid local control handshake");
  }
#else
  (void)record;
  (void)timeout_ms;
  Unsupported();
#endif
}

class EndpointLease::Impl {
public:
  Impl(std::string address, int value)
      : host(std::move(address)), port(value) {}
  std::string host;
  int port;
#ifdef _WIN32
  Handle mutex;
  bool acquired = false;
#endif
};
EndpointLease::EndpointLease(std::string host, int port)
    : impl_(std::make_unique<Impl>(std::move(host), port)) {}
EndpointLease::~EndpointLease() {
#ifdef _WIN32
  if (impl_->acquired)
    ReleaseMutex(impl_->mutex.Get());
#endif
}
void EndpointLease::Acquire() {
#ifdef _WIN32
  if (impl_->port < 0 || impl_->port > 65535)
    throw ControlError(2, "serve port is outside its valid range");
  if (impl_->acquired)
    throw ControlError(16, "endpoint lease is already held");
  const auto host = NormalHost(impl_->host);
  // Port zero requests an independent OS-assigned port, not one shared
  // endpoint. Its exclusive socket reservation is the start serialization.
  if (impl_->port == 0)
    return;
  std::uint64_t hash = 14695981039346656037ULL;
  for (const auto c : host + ":" + std::to_string(impl_->port)) {
    hash ^= static_cast<unsigned char>(c);
    hash *= 1099511628211ULL;
  }
  std::array<char, 17> encoded{};
  std::snprintf(encoded.data(), encoded.size(), "%016llx",
                static_cast<unsigned long long>(hash));
  OwnerSecurity security;
  const auto name = L"Local\\Bubo.Endpoint." + SidText(CurrentSid()) + L"." +
                    Wide(encoded.data());
  impl_->mutex.Reset(CreateMutexW(security.Attributes(), FALSE, name.c_str()));
  if (!impl_->mutex || !PrivateOwner(impl_->mutex.Get(), SE_KERNEL_OBJECT))
    throw ControlError(16, "cannot open private endpoint start mutex");
  const auto result = WaitForSingleObject(impl_->mutex.Get(), 0);
  if (result == WAIT_TIMEOUT)
    throw ControlError(13, "another CLI instance owns the requested endpoint");
  if (result != WAIT_OBJECT_0 && result != WAIT_ABANDONED)
    throw ControlError(16, "cannot acquire endpoint start mutex");
  impl_->acquired = true;
#else
  Unsupported();
#endif
}

class PortReservation::Impl {
public:
  Impl(std::string address, int value)
      : host(std::move(address)), port(value) {}
  std::string host;
  int port;
#ifdef _WIN32
  SOCKET socket = INVALID_SOCKET;
  bool winsock = false;
#endif
};
PortReservation::PortReservation(std::string host, int port)
    : impl_(std::make_unique<Impl>(std::move(host), port)) {}
PortReservation::~PortReservation() {
  Release();
}
int PortReservation::Port() const {
  return impl_->port;
}
void PortReservation::Reserve() {
#ifdef _WIN32
  if (impl_->socket != INVALID_SOCKET)
    throw ControlError(16, "port reservation is already held");
  if (impl_->port < 0 || impl_->port > 65535)
    throw ControlError(2, "serve port is outside its valid range");
  const auto host = NormalHost(impl_->host);
  WSADATA data{};
  if (WSAStartup(MAKEWORD(2, 2), &data) != 0)
    throw ControlError(1, "cannot initialize port reservation");
  impl_->winsock = true;
  sockaddr_storage address{};
  auto* ipv4 = reinterpret_cast<sockaddr_in*>(&address);
  auto* ipv6 = reinterpret_cast<sockaddr_in6*>(&address);
  int family = AF_INET;
  int address_size = sizeof(sockaddr_in);
  if (InetPtonA(AF_INET, host.c_str(), &ipv4->sin_addr) == 1) {
    ipv4->sin_family = AF_INET;
    ipv4->sin_port = htons(static_cast<unsigned short>(impl_->port));
  } else {
    family = AF_INET6;
    address_size = sizeof(sockaddr_in6);
    InetPtonA(AF_INET6, host.c_str(), &ipv6->sin6_addr);
    ipv6->sin6_family = AF_INET6;
    ipv6->sin6_port = htons(static_cast<unsigned short>(impl_->port));
  }
  impl_->socket = socket(family, SOCK_STREAM, IPPROTO_TCP);
  const int exclusive = 1;
  if (impl_->socket == INVALID_SOCKET ||
      setsockopt(impl_->socket, SOL_SOCKET, SO_EXCLUSIVEADDRUSE,
                 reinterpret_cast<const char*>(&exclusive),
                 sizeof(exclusive)) != 0 ||
      bind(impl_->socket, reinterpret_cast<const sockaddr*>(&address),
           address_size) != 0) {
    Release();
    throw ControlError(13, "requested endpoint cannot be reserved");
  }
  if (impl_->port == 0) {
    if (getsockname(impl_->socket, reinterpret_cast<sockaddr*>(&address),
                    &address_size) != 0) {
      Release();
      throw ControlError(13, "cannot identify reserved endpoint");
    }
    impl_->port =
        family == AF_INET ? ntohs(ipv4->sin_port) : ntohs(ipv6->sin6_port);
  }
#else
  Unsupported();
#endif
}
void PortReservation::Release() {
#ifdef _WIN32
  if (impl_->socket != INVALID_SOCKET) {
    closesocket(impl_->socket);
    impl_->socket = INVALID_SOCKET;
  }
  if (impl_->winsock) {
    WSACleanup();
    impl_->winsock = false;
  }
#endif
}
}  // namespace bubo::cli
