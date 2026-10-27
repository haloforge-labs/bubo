#include "src/cli/prepare/runtime_location.hpp"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <system_error>
#include <utility>

#include "src/cli/prepare/runtime_prepare.hpp"
#ifdef _WIN32
#include <windows.h>
#endif

namespace bubo::prepare {
namespace fs = std::filesystem;
namespace {
fs::path U8(std::string_view text) {
  return fs::path(std::u8string(reinterpret_cast<const char8_t*>(text.data()),
                                text.size()))
      .make_preferred();
}
std::string Narrow(const fs::path& path) {
  const auto text = path.u8string();
  return std::string(text.begin(), text.end());
}
std::string Key(const fs::path& path) {
  auto text = Narrow(path.lexically_normal());
  std::replace(text.begin(), text.end(), '/', '\\');
  while (text.size() > 3 && text.back() == '\\')
    text.pop_back();
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return text;
}
std::string ProcessEnvironment(std::string_view name) {
#ifdef _WIN32
  const std::wstring wide(name.begin(), name.end());
  const auto count = GetEnvironmentVariableW(wide.c_str(), nullptr, 0);
  if (!count)
    return {};
  std::wstring value(count, L'\0');
  const auto length =
      GetEnvironmentVariableW(wide.c_str(), value.data(), count);
  if (!length || length >= count)
    return {};
  value.resize(length);
  return Narrow(fs::path(value));
#else
  const std::string copy(name);
  const char* value = std::getenv(copy.c_str());
  return value ? value : "";
#endif
}
bool HasJournal(const fs::path& dir) {
  std::error_code ec;
  return fs::exists(dir / ".bubo" / "runtime" / "journal.json", ec);
}
}  // namespace

std::string_view RuntimeKindName(RuntimeKind kind) {
  switch (kind) {
    case RuntimeKind::Explicit:
      return "explicit";
    case RuntimeKind::Shared:
      return "shared";
    case RuntimeKind::Legacy:
      return "legacy";
  }
  return "shared";
}

fs::path SharedRuntimeDir(const fs::path& data_root) {
  return data_root / "runtime" / fs::path(std::string(kRuntimeId));
}
fs::path SharedRuntimeDirFor(std::string_view local_app_data) {
  if (local_app_data.empty())
    return {};
  return SharedRuntimeDir(U8(local_app_data) / "Bubo");
}

std::string CheckRuntimeDirOption(std::string_view text) {
  if (text.empty())
    return "the runtime folder is empty; use an absolute local folder";
  const auto path = U8(text);
  if (!path.is_absolute() || !path.has_root_directory())
    return "the runtime folder must be an absolute local path: " +
           std::string(text);
  const auto form = Narrow(path);
  const auto drive_path = [](std::string_view value) {
    return value.size() >= 3 &&
           ((value[0] >= 'A' && value[0] <= 'Z') ||
            (value[0] >= 'a' && value[0] <= 'z')) &&
           value[1] == ':' && value[2] == '\\';
  };
  // Only extended drive paths are supported. Accepting every \\?\ prefix
  // would also admit UNC and device namespaces before the readiness probe.
  // A rooted NT namespace is not a drive-qualified local folder either.
  const bool extended_drive = form.starts_with("\\\\?\\") &&
                              drive_path(std::string_view(form).substr(4));
#ifdef _WIN32
  if (!drive_path(form) && !extended_drive)
#else
  if (form.starts_with("\\\\") && !extended_drive)
#endif
    return "the runtime folder must be on a local drive: " + std::string(text);
  return {};
}

bool HasRuntimeMarker(const fs::path& dir) {
  std::error_code ec;
  return fs::exists(dir / ".bubo" / "runtime-ready.json", ec);
}

std::string DefaultRuntimeCheck(const fs::path& dir) {
  std::error_code ec;
  if (HasRuntimeMarker(dir) || HasJournal(dir))
    return RuntimeReadyError(dir);
  if (fs::is_regular_file(dir / "amdhip64_7.dll", ec))
    return {};
  return "no AMD runtime found in " + Narrow(dir) +
         "; run bubo prepare runtime";
}

RuntimeLocation ResolveRuntime(const RuntimeSearch& search) {
  const auto check = search.check ? search.check : DefaultRuntimeCheck;
  RuntimeLocation result;
  const std::pair<const std::string*, const char*> explicit_choices[] = {
      {&search.option, "option"}, {&search.environment, "environment"}};
  for (const auto& [text, source] : explicit_choices) {
    if (text->empty())
      continue;
    result.dir = U8(*text);
    result.kind = RuntimeKind::Explicit;
    result.source = source;
    result.error = CheckRuntimeDirOption(*text);
    if (result.error.empty())
      result.error = check(result.dir);
    return result;
  }
  const auto shared = SharedRuntimeDirFor(search.local_app_data);
  if (!shared.empty() && HasRuntimeMarker(shared) && check(shared).empty()) {
    result.dir = shared;
    result.kind = RuntimeKind::Shared;
    result.source = "shared";
    return result;
  }
  const auto& legacy = search.exe_dir;
  if (!legacy.empty() && check(legacy).empty()) {
    result.dir = legacy;
    result.kind = RuntimeKind::Legacy;
    result.source = "legacy";
    return result;
  }
  // Nothing usable. Name the folder the user most likely meant: a damaged
  // legacy install (marker or journal present) when the shared folder has
  // none, else the shared folder.
  const bool shared_started =
      !shared.empty() && (HasRuntimeMarker(shared) || HasJournal(shared));
  if (!legacy.empty() && !shared_started &&
      (HasRuntimeMarker(legacy) || HasJournal(legacy))) {
    result.dir = legacy;
    result.kind = RuntimeKind::Legacy;
    result.source = "legacy";
    result.error = check(legacy);
    return result;
  }
  result.dir = shared.empty() ? legacy : shared;
  result.kind = shared.empty() ? RuntimeKind::Legacy : RuntimeKind::Shared;
  result.source = "default";
  result.absent = !shared_started;
  result.error = check(result.dir);
  if (result.error.empty())
    result.error = "no AMD runtime found in " + Narrow(result.dir) +
                   "; run bubo prepare runtime";
  return result;
}

std::string RuntimeDirFromEnvironment() {
  return ProcessEnvironment(kRuntimeDirEnvironment);
}

RuntimeSearch ProcessRuntimeSearch(std::string_view local_app_data,
                                   const fs::path& exe_dir) {
  RuntimeSearch search;
  search.environment = RuntimeDirFromEnvironment();
  search.local_app_data = std::string(local_app_data);
  search.exe_dir = exe_dir;
  return search;
}

RuntimeLocation ResolveProcessRuntime(std::string_view local_app_data,
                                      const fs::path& exe_dir) {
  return ResolveRuntime(ProcessRuntimeSearch(local_app_data, exe_dir));
}

std::string_view ClassifyRuntime(const fs::path& runtime,
                                 std::string_view local_app_data) {
  const auto shared = SharedRuntimeDirFor(local_app_data);
  if (!shared.empty() && Key(shared) == Key(runtime))
    return "shared";
  const auto env = ProcessEnvironment(kRuntimeDirEnvironment);
  if (!env.empty() && Key(U8(env)) == Key(runtime))
    return "explicit";
  return "legacy";
}
}  // namespace bubo::prepare
