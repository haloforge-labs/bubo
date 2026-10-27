#include "src/cli/support_bundle.hpp"

#include <zlib.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <initializer_list>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include "src/cli/arg_parser.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#endif

#include "src/cli/brand.hpp"
#include "src/cli/machine_output.hpp"
#include "src/cli/prepare/runtime_location.hpp"
#include "src/cli/status.hpp"
#include "src/cli/switch_listing.hpp"
#include "src/core/crypto/sha256.hpp"
#include "src/core/prepare/hash_file.hpp"

namespace bubo::cli {
namespace {
namespace fs = std::filesystem;
using gufo::json::Value;

constexpr std::array<std::string_view, 7> kNames = {
    "versions.json", "settings.json", "status.json",  "diagnose.json",
    "history.json",  "logs.json",     "manifest.json"};
constexpr std::size_t kMaxMemberBytes = 32 * 1024;
constexpr std::size_t kMaxTextBytes = 128 * 1024;
constexpr std::size_t kMaxZipBytes = 160 * 1024;
constexpr std::size_t kMaxSourceJsonBytes = 64 * 1024;
constexpr std::size_t kMaxLogTailBytes = 32 * 1024;
constexpr std::size_t kMaxLogLines = 256;
constexpr std::size_t kMaxHistory = 32;

std::string Hash(std::string_view bytes) {
  return gufo::crypto::Sha256Hex(std::span(
      reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size()));
}

bool OneOf(std::string_view value,
           std::initializer_list<std::string_view> allowed) {
  return std::find(allowed.begin(), allowed.end(), value) != allowed.end();
}

void ProjectEnum(Value& target, std::string_view key, const Value& source,
                 std::initializer_list<std::string_view> allowed) {
  const auto* field = source.find(std::string(key));
  if (field && field->is_string() && OneOf(field->str(), allowed))
    target[std::string(key)] = field->str();
}

void ProjectBool(Value& target, std::string_view key, const Value& source) {
  const auto* field = source.find(std::string(key));
  if (field && field->is_bool())
    target[std::string(key)] = field->as_bool();
}

void ProjectInteger(Value& target, std::string_view key, const Value& source,
                    double minimum, double maximum) {
  const auto* field = source.find(std::string(key));
  if (!field || !field->is_number())
    return;
  const double number = field->as_double();
  if (std::isfinite(number) && number >= minimum && number <= maximum &&
      std::floor(number) == number)
    target[std::string(key)] = Value(number);
}

void ProjectErrorCode(Value& target, const Value& source) {
  const auto* error = source.find("last_error");
  if (!error || !error->is_object())
    return;
  const auto* code = error->find("code");
  if (code && code->is_string() &&
      OneOf(code->str(),
            {"unexpected_failure", "invalid_arguments", "runtime_missing",
             "model_missing", "port_occupied", "startup_timeout",
             "protocol_mismatch", "control_access", "stop_timeout",
             "unsupported", "device_lost", "diagnostic_failed",
             "storage_failed", "cancelled", "integrity_failed",
             "memory_insufficient"}))
    target["last_error_code"] = code->str();
}

Value ProjectSettings(const Value& input) {
  auto result = Value::object();
  result["available"] = input.is_object();
  if (!input.is_object())
    return result;
  ProjectInteger(result, "version", input, 1, 1);
  ProjectInteger(result, "port", input, 1, 65535);
  ProjectInteger(result, "context", input, 1, 1'048'576);
  ProjectInteger(result, "sessions", input, 1, 64);
  for (const auto* key :
       {"start_gui_at_login", "start_server_with_gui", "close_to_tray"})
    ProjectBool(result, key, input);
  ProjectEnum(result, "profile", input,
              {"default", "exact", "vanilla", "stock"});
  ProjectEnum(result, "npu_prefill", input, {"on", "off"});
  ProjectEnum(result, "effort", input,
              {"auto", "minimal", "low", "medium", "high", "xhigh", "max"});
  ProjectErrorCode(result, input);
  return result;
}

Value ProjectEffectiveSettings(const Value& input) {
  auto result = Value::object();
  result["available"] = input.is_object();
  if (!input.is_object())
    return result;
  ProjectEnum(result, "profile", input,
              {"default", "exact", "vanilla", "stock", "subagents"});
  auto rows = Value::array();
  const auto* actual = input.find("effective_switches");
  if (!actual || !actual->is_array())
    return result;
  const auto listing = SwitchListing();
  for (const auto& catalog : listing.find("switches")->items()) {
    const auto name = catalog.member_str("name");
    if (name == "--api-key" || name == "BUBO_API_KEY" ||
        (catalog.member_str("kind") == "env" &&
         !catalog.find("echo")->as_bool()))
      continue;
    const auto found = std::find_if(
        actual->items().begin(), actual->items().end(),
        [&](const auto& item) { return item.member_str("name") == name; });
    if (found == actual->items().end())
      continue;
    const auto* effective = found->find("effective");
    if (!effective)
      continue;
    auto row = Value::object();
    row["name"] = name;
    row["effective"] = Value();
    const auto& allowed = *catalog.find("allowed");
    if (const auto* values = allowed.find("values");
        values && values->is_array()) {
      for (const auto& permitted : values->items())
        if (effective->dump() == permitted.dump())
          row["effective"] = permitted;
    } else if (allowed.member_str("type") == "integer" ||
               allowed.member_str("type") == "number" ||
               allowed.member_str("type") == "boolean") {
      double number =
          effective->as_double(std::numeric_limits<double>::quiet_NaN());
      if (effective->is_string()) {
        const auto& text = effective->str();
        const auto parsed =
            std::from_chars(text.data(), text.data() + text.size(), number);
        if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size())
          number = std::numeric_limits<double>::quiet_NaN();
      }
      if (std::isfinite(number) && std::abs(number) <= 9007199254740991.0)
        row["effective"] = number;
      else if (effective->is_bool())
        row["effective"] = effective->as_bool();
    }
    ProjectEnum(row, "source", *found, {"default", "profile", "user", "env"});
    row["constrained"] =
        found->find("reason") && !found->find("reason")->is_null();
    rows.push_back(std::move(row));
  }
  result["effective_switches"] = std::move(rows);
  return result;
}

Value ProjectStatus(const Value& input) {
  auto result = Value::object();
  const auto* snapshot_available = input.find("support_snapshot_available");
  result["available"] = input.is_object() && (!snapshot_available ||
                                              !snapshot_available->is_bool() ||
                                              snapshot_available->as_bool());
  if (!input.is_object())
    return result;
  ProjectBool(result, "prepared", input);
  ProjectBool(result, "selection_required", input);
  ProjectEnum(result, "phase", input,
              {"not_prepared", "stopped", "starting", "loading", "ready",
               "stopping", "error", "multiple_instances"});
  if (const auto* readiness = input.find("readiness");
      readiness && readiness->is_object()) {
    auto safe = Value::object();
    for (const auto* key : {"runtime", "model", "server"})
      ProjectBool(safe, key, *readiness);
    result["readiness"] = std::move(safe);
  }
  if (const auto* counters = input.find("counters");
      counters && counters->is_object()) {
    auto safe = Value::object();
    for (const auto* key : {"prefill_tokens_total", "decode_tokens_total",
                            "completed_requests_total", "active_requests"})
      ProjectInteger(safe, key, *counters, 0, 1e15);
    result["counters"] = std::move(safe);
  }
  ProjectErrorCode(result, input);
  return result;
}

Value ProjectDiagnose(const Value& input) {
  auto result = Value::object();
  result["available"] = input.is_object();
  if (!input.is_object())
    return result;
  ProjectEnum(result, "source", input, {"persisted_state", "status_readiness"});
  ProjectEnum(result, "status", input, {"PASS", "WARN", "FAIL"});
  auto checks = Value::array();
  const auto* raw = input.find("checks");
  if (raw && raw->is_array()) {
    const auto count = std::min<std::size_t>(raw->size(), 32);
    for (std::size_t index = 0; index < count; ++index) {
      const auto& check = raw->items()[index];
      if (!check.is_object())
        continue;
      auto safe = Value::object();
      ProjectEnum(safe, "name", check,
                  {"platform", "system", "cpu", "memory", "gpu", "toolchain",
                   "runtime", "model"});
      ProjectEnum(safe, "status", check, {"PASS", "WARN", "FAIL"});
      if (safe.size() == 2)
        checks.push_back(std::move(safe));
    }
  }
  result["checks"] = std::move(checks);
  return result;
}

Value ReadinessDiagnose(const Value& status) {
  auto result = Value::object();
  const auto* readiness = status.find("readiness");
  if (!readiness || !readiness->is_object())
    return {};
  auto checks = Value::array();
  bool all_ready = true;
  for (const auto* name : {"runtime", "model"}) {
    const auto* ready = readiness->find(name);
    if (!ready || !ready->is_bool())
      return {};
    auto row = Value::object();
    row["name"] = name;
    row["status"] = ready->as_bool() ? "PASS" : "WARN";
    all_ready &= ready->as_bool();
    checks.push_back(std::move(row));
  }
  result["source"] = "status_readiness";
  result["status"] = all_ready ? "PASS" : "WARN";
  result["checks"] = std::move(checks);
  return result;
}

Value ProjectHistory(const std::vector<Value>& history) {
  auto result = Value::array();
  const auto start =
      history.size() > kMaxHistory ? history.size() - kMaxHistory : 0;
  for (std::size_t index = start; index < history.size(); ++index) {
    const auto& input = history[index];
    if (!input.is_object())
      continue;
    auto row = Value::object();
    ProjectEnum(row, "type", input,
                {"diagnose", "status", "starting", "loading", "ready",
                 "stopping", "stopped", "error"});
    if (!row.contains("type"))
      continue;
    const auto* payload = input.find("payload");
    const auto& state = payload && payload->is_object() ? *payload : input;
    if (row.member_str("type") == "diagnose") {
      row["state"] = ProjectDiagnose(state);
    } else {
      row["state"] = ProjectStatus(state);
    }
    result.push_back(std::move(row));
  }
  return result;
}

Value ProjectLogs(const std::vector<std::string>& logs) {
  auto result = Value::object();
  auto rows = Value::array();
  std::size_t dropped = 0;
  const auto start =
      logs.size() > kMaxLogLines ? logs.size() - kMaxLogLines : 0;
  for (std::size_t index = start; index < logs.size(); ++index) {
    const auto& line = logs[index];
    if (line.size() > 4096) {
      ++dropped;
      continue;
    }
    try {
      const auto raw = gufo::json::parse(line);
      if (!raw.is_object()) {
        ++dropped;
        continue;
      }
      auto row = Value::object();
      ProjectEnum(row, "event", raw,
                  {"startup", "ready", "stopping", "stopped", "error",
                   "diagnose", "log_drop"});
      ProjectEnum(row, "level", raw, {"debug", "info", "warn", "error"});
      ProjectInteger(row, "count", raw, 0, 1'000'000);
      ProjectInteger(row, "duration_ms", raw, 0, 86'400'000);
      if (!row.contains("event")) {
        ++dropped;
        continue;
      }
      rows.push_back(std::move(row));
    } catch (const std::exception&) {
      // Free-form, malformed, and unknown raw log lines are dropped whole.
      ++dropped;
    }
  }
  result["entries"] = std::move(rows);
  result["dropped_lines"] = static_cast<unsigned long long>(dropped);
  return result;
}

std::string SafeVersion(std::string_view input) {
  if (input.empty() || input.size() > 64)
    return "unknown";
  for (const unsigned char byte : input)
    if (!((byte >= '0' && byte <= '9') || (byte >= 'A' && byte <= 'Z') ||
          (byte >= 'a' && byte <= 'z') || byte == '.' || byte == '-' ||
          byte == '+'))
      return "unknown";
  return std::string(input);
}

std::string SafeRevision(std::string_view input) {
  if (input.size() < 7 || input.size() > 64)
    return "unknown";
  for (const char byte : input)
    if (!((byte >= '0' && byte <= '9') || (byte >= 'a' && byte <= 'f')))
      return "unknown";
  return std::string(input);
}

std::string JsonText(const Value& value) {
  return value.dump() + '\n';
}

void Append16(std::string& bytes, std::uint16_t value) {
  bytes.push_back(static_cast<char>(value & 0xff));
  bytes.push_back(static_cast<char>((value >> 8) & 0xff));
}

void Append32(std::string& bytes, std::uint32_t value) {
  Append16(bytes, static_cast<std::uint16_t>(value & 0xffff));
  Append16(bytes, static_cast<std::uint16_t>(value >> 16));
}

std::string SerializeZip(const std::vector<SupportBundleFile>& files) {
  if (files.size() != kNames.size())
    throw std::length_error("invalid support bundle member count");
  std::size_t text_bytes = 0;
  std::string result, directory;
  for (std::size_t index = 0; index < files.size(); ++index) {
    const auto& file = files[index];
    if (file.name != kNames[index] || file.text.size() > kMaxMemberBytes ||
        file.sha256 != Hash(file.text) ||
        text_bytes + file.text.size() > kMaxTextBytes)
      throw std::invalid_argument("invalid support bundle member");
    text_bytes += file.text.size();
    const auto offset = static_cast<std::uint32_t>(result.size());
    const auto size = static_cast<std::uint32_t>(file.text.size());
    const auto crc = static_cast<std::uint32_t>(crc32(
        crc32(0L, Z_NULL, 0), reinterpret_cast<const Bytef*>(file.text.data()),
        static_cast<uInt>(file.text.size())));
    Append32(result, 0x04034b50);
    Append16(result, 20);
    Append16(result, 0);
    Append16(result, 0);  // Stored: never compress untrusted source bytes.
    Append16(result, 0);
    Append16(result, 0x0021);  // Fixed DOS date: 1980-01-01.
    Append32(result, crc);
    Append32(result, size);
    Append32(result, size);
    Append16(result, static_cast<std::uint16_t>(file.name.size()));
    Append16(result, 0);
    result += file.name;
    result += file.text;

    Append32(directory, 0x02014b50);
    Append16(directory, 20);
    Append16(directory, 20);
    Append16(directory, 0);
    Append16(directory, 0);
    Append16(directory, 0);
    Append16(directory, 0x0021);
    Append32(directory, crc);
    Append32(directory, size);
    Append32(directory, size);
    Append16(directory, static_cast<std::uint16_t>(file.name.size()));
    Append16(directory, 0);
    Append16(directory, 0);
    Append16(directory, 0);
    Append16(directory, 0);
    Append32(directory, 0);
    Append32(directory, offset);
    directory += file.name;
  }
  const auto directory_offset = static_cast<std::uint32_t>(result.size());
  result += directory;
  Append32(result, 0x06054b50);
  Append16(result, 0);
  Append16(result, 0);
  Append16(result, static_cast<std::uint16_t>(files.size()));
  Append16(result, static_cast<std::uint16_t>(files.size()));
  Append32(result, static_cast<std::uint32_t>(directory.size()));
  Append32(result, directory_offset);
  Append16(result, 0);
  if (result.size() > kMaxZipBytes)
    throw std::length_error("support bundle exceeds ZIP size limit");
  return result;
}

void ValidateBundle(const SupportBundle& bundle) {
  if (bundle.zip.size() > kMaxZipBytes ||
      bundle.zip != SerializeZip(bundle.files) ||
      bundle.zip_sha256 != Hash(bundle.zip))
    throw std::invalid_argument("invalid support bundle image");
}

Value ReadJson(const fs::path& path) {
  std::error_code ec;
  if (fs::is_symlink(fs::symlink_status(path, ec)) || ec ||
      !fs::is_regular_file(path, ec) || ec)
    return {};
  const auto size = fs::file_size(path, ec);
  if (ec || size > kMaxSourceJsonBytes)
    return {};
  std::ifstream input(path, std::ios::binary);
  if (!input)
    return {};
  std::string text(static_cast<std::size_t>(size), '\0');
  input.read(text.data(), static_cast<std::streamsize>(text.size()));
  if (static_cast<std::size_t>(input.gcount()) != text.size())
    return {};
  try {
    return gufo::json::parse(text);
  } catch (const std::exception&) {
    return {};
  }
}

void AppendLogTail(const fs::path& path, std::vector<std::string>& lines) {
  std::error_code ec;
  if (fs::is_symlink(fs::symlink_status(path, ec)) || ec ||
      !fs::is_regular_file(path, ec) || ec)
    return;
  const auto size = fs::file_size(path, ec);
  if (ec || size > static_cast<std::uintmax_t>(
                       std::numeric_limits<std::streamoff>::max()))
    return;
  std::ifstream input(path, std::ios::binary);
  if (!input)
    return;
  const auto length = static_cast<std::size_t>(
      std::min<std::uintmax_t>(size, kMaxLogTailBytes));
  if (size > length)
    input.seekg(static_cast<std::streamoff>(size - length));
  std::string tail(length, '\0');
  input.read(tail.data(), static_cast<std::streamsize>(length));
  tail.resize(static_cast<std::size_t>(input.gcount()));
  std::size_t start = 0;
  if (size > length) {
    const auto newline = tail.find('\n');
    start = newline == std::string::npos ? tail.size() : newline + 1;
  }
  while (start < tail.size()) {
    const auto newline = tail.find('\n', start);
    if (newline == std::string::npos)
      break;  // A partial final record is never interpreted.
    if (newline - start <= 4096)
      lines.emplace_back(tail.substr(start, newline - start));
    start = newline + 1;
  }
}

fs::path SourceRoot(std::string_view local_app_data) {
  if (local_app_data.empty() || local_app_data.size() > 4096)
    return {};
  const auto root = fs::u8path(std::string(local_app_data));
  if (!root.is_absolute())
    return {};
  for (const auto& part : root)
    if (part == "..")
      return {};
  return root / "Bubo" / "gui";
}

bool ReparsePoint(const fs::path& path) {
#ifdef _WIN32
  const auto attributes = GetFileAttributesW(path.c_str());
  return attributes == INVALID_FILE_ATTRIBUTES ||
         (attributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0;
#else
  std::error_code ec;
  return fs::is_symlink(fs::symlink_status(path, ec)) || ec;
#endif
}

void CheckTargetSyntax(const fs::path& target) {
  if (target.empty() || !target.is_absolute() || target.extension() != ".zip" ||
      target.native().size() > 4096)
    throw std::invalid_argument("invalid support bundle path");
  for (const auto& part : target)
    if (part == ".." || part == ".")
      throw std::invalid_argument("support bundle path traversal");
#ifdef _WIN32
  if (target.native().starts_with(L"\\\\"))
    throw std::invalid_argument("network and device paths are unsupported");
  for (const auto& part : target) {
    if (part == target.root_name() || part == target.root_directory())
      continue;
    const auto& name = part.native();
    if (name.empty() || name.find(L':') != std::wstring::npos ||
        name.back() == L'.' || name.back() == L' ')
      throw std::invalid_argument("unsafe Windows path component");
    std::wstring stem = name.substr(0, name.find(L'.'));
    for (auto& letter : stem)
      if (letter >= L'a' && letter <= L'z')
        letter -= L'a' - L'A';
    if (stem == L"CON" || stem == L"PRN" || stem == L"AUX" || stem == L"NUL" ||
        (stem.size() == 4 && stem[3] >= L'1' && stem[3] <= L'9' &&
         (stem.substr(0, 3) == L"COM" || stem.substr(0, 3) == L"LPT")))
      throw std::invalid_argument("reserved Windows path component");
  }
#else
  if (target.native().starts_with("//"))
    throw std::invalid_argument("network paths are unsupported");
#endif
}

fs::path CheckedTarget(const fs::path& target) {
  CheckTargetSyntax(target);
  std::error_code ec;
  const auto absolute = fs::absolute(target, ec);
  if (ec || absolute.empty() || absolute.filename().empty())
    throw std::invalid_argument("invalid support bundle path");
  auto parent = absolute.parent_path();
  if (!fs::is_directory(parent, ec) || ec)
    throw std::runtime_error("support bundle parent is unavailable");
  for (auto path = parent; !path.empty(); path = path.parent_path()) {
    if (ReparsePoint(path))
      throw std::runtime_error("support bundle path has a link");
    if (path == path.root_path())
      break;
  }
  if (fs::exists(absolute, ec) || ec)
    throw std::runtime_error("support bundle target already exists");
  return absolute;
}

void WriteNewFile(const fs::path& path, std::string_view bytes) {
#ifdef _WIN32
  const HANDLE output = CreateFileW(path.c_str(), GENERIC_WRITE, 0, nullptr,
                                    CREATE_NEW, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (output == INVALID_HANDLE_VALUE)
    throw std::runtime_error("cannot create support bundle");
  bool ok = true;
  while (!bytes.empty() && ok) {
    DWORD written = 0;
    ok = WriteFile(output, bytes.data(), static_cast<DWORD>(bytes.size()),
                   &written, nullptr) &&
         written != 0;
    bytes.remove_prefix(written);
  }
  ok = ok && FlushFileBuffers(output);
  ok = CloseHandle(output) && ok;
#else
  const int output = open(
      path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
  if (output < 0)
    throw std::runtime_error("cannot create support bundle");
  bool ok = true;
  while (!bytes.empty() && ok) {
    const auto written = write(output, bytes.data(), bytes.size());
    if (written < 0 && errno == EINTR)
      continue;
    ok = written > 0;
    if (ok)
      bytes.remove_prefix(static_cast<std::size_t>(written));
  }
  ok = ok && fsync(output) == 0;
  ok = close(output) == 0 && ok;
#endif
  if (!ok) {
    std::error_code ec;
    fs::remove(path, ec);
    throw std::runtime_error("cannot finish support bundle");
  }
}

}  // namespace

SupportBundle BuildSupportBundle(const Value& status,
                                 const std::vector<Value>& history,
                                 const std::vector<std::string>& logs,
                                 const SupportBinaryHashes& binaries) {
  SupportBundle bundle;
  auto versions = Value::object();
  versions["schema_version"] = 1;
  versions["product"] = "bubo";
  versions["product_version"] = SafeVersion(gufo::cli::kProductVersion);
  versions["engine_version"] = SafeVersion(gufo::cli::kEngineVersion);
  versions["revision"] = SafeRevision(gufo::cli::kRevision);
  auto hashes = Value::object();
  const auto hash_value = [](const std::optional<std::string>& hash) {
    if (!hash || hash->size() != 64 ||
        !std::all_of(hash->begin(), hash->end(), [](char c) {
          return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
        }))
      return Value();
    return Value(*hash);
  };
  hashes["bootstrap"] = hash_value(binaries.bootstrap);
  hashes["engine"] = hash_value(binaries.engine);
  versions["binary_sha256"] = std::move(hashes);

  const auto* settings = status.find("effective_settings");
  const auto* gui_settings = status.find("gui_settings");
  const auto* diagnose = status.find("diagnose_state");
  const std::array<Value, 6> documents = {
      std::move(versions),
      settings && settings->find("effective_switches") &&
              settings->find("effective_switches")->is_array()
          ? ProjectEffectiveSettings(*settings)
          : ProjectSettings(gui_settings ? *gui_settings : Value()),
      ProjectStatus(status),
      ProjectDiagnose(diagnose ? *diagnose : Value()),
      ProjectHistory(history),
      ProjectLogs(logs)};
  for (std::size_t index = 0; index < documents.size(); ++index) {
    auto text = JsonText(documents[index]);
    bundle.files.push_back({std::string(kNames[index]), std::move(text), {}});
    bundle.files.back().sha256 = Hash(bundle.files.back().text);
  }
  auto manifest = Value::object();
  manifest["schema_version"] = 1;
  auto members = Value::array();
  for (const auto& file : bundle.files) {
    auto item = Value::object();
    item["name"] = file.name;
    item["bytes"] = static_cast<unsigned long long>(file.text.size());
    item["sha256"] = file.sha256;
    members.push_back(std::move(item));
  }
  manifest["files"] = std::move(members);
  bundle.files.push_back({std::string(kNames.back()), JsonText(manifest), {}});
  bundle.files.back().sha256 = Hash(bundle.files.back().text);
  bundle.zip = SerializeZip(bundle.files);
  bundle.zip_sha256 = Hash(bundle.zip);
  return bundle;
}

Value SupportBundleEvent(const SupportBundle& bundle, bool saved) {
  ValidateBundle(bundle);
  auto event = Value::object();
  event["schema_version"] = 1;
  event["saved"] = saved;
  event["zip_bytes"] = static_cast<unsigned long long>(bundle.zip.size());
  event["bundle_sha256"] = bundle.zip_sha256;
  auto files = Value::array();
  auto preview = Value::object();
  auto hashes = Value::object();
  for (const auto& file : bundle.files) {
    files.push_back(file.name);
    preview[file.name] = file.text;
    hashes[file.name] = file.sha256;
  }
  event["files"] = std::move(files);
  event["text_preview"] = std::move(preview);
  event["sha256"] = std::move(hashes);
  return event;
}

void SaveSupportBundle(const SupportBundle& bundle, const fs::path& path) {
  ValidateBundle(bundle);
  WriteNewFile(CheckedTarget(path), bundle.zip);
}

bool ParseSupportBundleOptions(std::span<const char* const> args,
                               SupportBundleOptions& options) {
  gufo::cli::ArgParser parser("bubo diagnose");
  parser.SetImplicitHelp(false);
  const auto once = [&](std::string_view flag, bool* target) {
    parser.AddCustomFlag("", flag, "Bundle option", "Support",
                         [target](auto, auto, auto*) {
                           if (*target)
                             return false;
                           *target = true;
                           return true;
                         });
  };
  once("--json", &options.json);
  once("--preview", &options.preview);
  parser.AddCustomOption("", "--bundle", "PATH", "Save bundle", "Support",
                         [&](auto, auto value, auto*) {
                           if (options.bundle_option)
                             return false;
                           options.bundle_option = true;
                           options.path_text = value;
                           return true;
                         });
  if (!parser.Parse(args))
    return false;
  return options.json && (options.preview || options.bundle_option) &&
         (!options.bundle_option ||
          (!options.path_text.empty() && !options.path_text.starts_with("--") &&
           options.path_text.size() <= 4096));
}

int RunSupportBundle(std::span<const char* const> args,
                     std::string_view local_app_data, const fs::path& runtime,
                     std::ostream& out, std::ostream& err) {
  (void)err;  // Diagnostics and paths must never leak through raw stderr.
  MachineWriter writer(out);
  SupportBundleOptions options;
  if (!ParseSupportBundleOptions(args, options)) {
    writer.Write("error",
                 Error("invalid_arguments", "diagnose", "bundle", false,
                       "Use diagnose --json --preview or --bundle PATH."));
    return 2;
  }
  const auto& path_text = options.path_text;
  const auto bundle_option = options.bundle_option;
  const auto preview = options.preview;
  try {
    // Syntax is checked even in preview mode, without touching the target.
    if (bundle_option)
      CheckTargetSyntax(fs::u8path(path_text));
    Value status;
    try {
      const std::array<const char*, 1> status_args = {"--json"};
      std::ostringstream status_out, status_err;
      // `runtime` is the bubo.exe folder here; the AMD runtime may be in the
      // shared per-user folder.
      const auto amd_runtime =
          bubo::prepare::ResolveProcessRuntime(local_app_data, runtime).dir;
      if (RunStatus(status_args, local_app_data, amd_runtime, status_out,
                    status_err) == 0)
        status = gufo::json::parse(status_out.str())["payload"];
      else
        status = StatusSnapshot(local_app_data, amd_runtime);
    } catch (const std::exception&) {
      status = Value::object();
      status["support_snapshot_available"] = false;
    }
    std::vector<Value> history;
    std::vector<std::string> logs;
    if (const auto* transitions = status.find("state_history");
        transitions && transitions->is_array())
      for (const auto& row : transitions->items())
        history.push_back(row);
    if (const auto* live_log = status.find("log_path");
        live_log && live_log->is_string())
      AppendLogTail(fs::u8path(live_log->str()), logs);
    const auto source = SourceRoot(local_app_data);
    if (!source.empty()) {
      status["gui_settings"] = ReadJson(source / "settings.json");
      auto diagnose = ReadJson(source / "diagnose-state.json");
      if (diagnose.is_object())
        diagnose["source"] = "persisted_state";
      else
        diagnose = ReadinessDiagnose(status);
      status["diagnose_state"] = std::move(diagnose);
      const auto raw_history = ReadJson(source / "diagnose-history.json");
      if (raw_history.is_array()) {
        const auto start = raw_history.size() > kMaxHistory
                               ? raw_history.size() - kMaxHistory
                               : 0;
        for (std::size_t index = start; index < raw_history.size(); ++index)
          history.push_back(raw_history.items()[index]);
      }
      for (const auto* name : {"stderr.2.log", "stderr.1.log", "stderr.0.log"})
        AppendLogTail(source / "logs" / name, logs);
    } else {
      status["diagnose_state"] = ReadinessDiagnose(status);
    }
    const auto binary_hash =
        [&](const char* name) -> std::optional<std::string> {
      try {
        const auto file = runtime / name;
        std::error_code ec;
        if (fs::is_symlink(fs::symlink_status(file, ec)) || ec ||
            !fs::is_regular_file(file, ec) || ec ||
            fs::file_size(file, ec) > 512ULL * 1024 * 1024 || ec)
          return {};
        return bubo::prepare::HashFile(file);
      } catch (const std::exception&) {
        return {};
      }
    };
    const SupportBinaryHashes binaries{binary_hash("bubo.exe"),
                                       binary_hash("bubo-engine.dll")};
    const auto bundle = BuildSupportBundle(status, history, logs, binaries);
    if (!preview)
      SaveSupportBundle(bundle, fs::u8path(path_text));
    if (!writer.Write("support_bundle", SupportBundleEvent(bundle, !preview)))
      return 1;
    return 0;
  } catch (const std::invalid_argument&) {
    writer.Write("error", Error("invalid_arguments", "diagnose", "bundle",
                                false, "Select a new local ZIP path."));
    return 2;
  } catch (const std::exception&) {
    writer.Write("error", Error("storage_failed", "diagnose", "bundle", false,
                                "Select a writable new ZIP path."));
    return 7;
  }
}
}  // namespace bubo::cli
