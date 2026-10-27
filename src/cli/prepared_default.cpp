#include "src/cli/prepared_default.hpp"

#include <cstdlib>
#include <stdexcept>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "src/core/prepare/file_ops.hpp"
#include "src/core/prepare/metadata_json.hpp"
#include "src/core/prepare/release_catalogue.hpp"

namespace bubo::cli {
namespace {
namespace fs = std::filesystem;
namespace prep = bubo::prepare;
std::string Utf8(const fs::path& path) {
  const auto text = path.u8string();
  return {text.begin(), text.end()};
}
fs::path Pointer(std::string_view local) {
  return local.empty() ? fs::path{}
                       : fs::u8path(local) / "Bubo/prepared-default.json";
}
fs::path ManifestPath(const fs::path& path) {
  const auto resolved =
      fs::is_directory(path) ? path / "bubo-package.json" : path;
  return fs::absolute(resolved).lexically_normal();
}
}  // namespace
std::string EnvironmentUtf8(std::string_view name) {
#ifdef _WIN32
  const std::wstring wide(name.begin(), name.end());
  const auto size = GetEnvironmentVariableW(wide.c_str(), nullptr, 0);
  // A present empty value needs only its terminator. A second read returns
  // zero for it too, so handle it before the changed-value check below.
  if (size <= 1)
    return {};
  std::wstring value(size, L'\0');
  const auto length = GetEnvironmentVariableW(wide.c_str(), value.data(), size);
  if (!length || length >= size)
    throw std::runtime_error("environment changed while reading");
  value.resize(length);
  return Utf8(fs::path(value));
#else
  const auto* value = std::getenv(std::string(name).c_str());
  return value ? value : "";
#endif
}
fs::path ExecutableDirectory() {
#ifdef _WIN32
  std::wstring name(32768, L'\0');
  const auto size =
      GetModuleFileNameW(nullptr, name.data(), static_cast<DWORD>(name.size()));
  if (!size || size == name.size())
    throw std::runtime_error("cannot locate executable");
  name.resize(size);
  return fs::path(name).parent_path();
#else
  return fs::read_symlink("/proc/self/exe").parent_path();
#endif
}
ModelSelection ResolvePreparedDefault(const fs::path& model,
                                      const fs::path& explicit_manifest,
                                      std::string_view environment_manifest,
                                      std::string_view local) {
  ModelSelection result;
  result.pointer = Pointer(local);
  result.model = model;
  if (!explicit_manifest.empty()) {
    result.manifest = ManifestPath(explicit_manifest);
    result.source = "explicit_manifest";
  } else if (!model.empty()) {
    result.source = "explicit_model";
  } else if (!environment_manifest.empty()) {
    result.manifest = ManifestPath(fs::u8path(environment_manifest));
    result.source = "environment";
  } else if (!result.pointer.empty() && fs::exists(result.pointer)) {
    const auto value =
        prep::metadata::Parse(prep::metadata::Read(result.pointer));
    prep::metadata::Keys(value, {"schema_version", "manifest_path"});
    if (prep::metadata::Integer(
            prep::metadata::Required(value, "schema_version")) != 1)
      throw std::invalid_argument(
          "prepared-default schema_version is unsupported");
    const auto saved = fs::u8path(prep::metadata::String(
        prep::metadata::Required(value, "manifest_path")));
    if (!saved.is_absolute() || saved.filename() != "bubo-package.json")
      throw std::invalid_argument(
          "prepared-default manifest_path must be an absolute "
          "bubo-package.json path");
    result.manifest = saved.lexically_normal();
    result.source = "prepared_default";
  } else {
    if (!local.empty())
      result.manifest = fs::u8path(local) / "Bubo/models" / prep::kPackageId /
                        "bubo-package.json";
    result.source = "standard_location";
  }
  // A missing or damaged saved selection is authoritative. Never scan other
  // disks or hide a broken selection behind a standard-location fallback.
  return result;
}
gufo::json::Value SelectionJson(const ModelSelection& selection) {
  auto value = gufo::json::Value::object();
  value["model"] = selection.model.empty()
                       ? gufo::json::Value()
                       : gufo::json::Value(Utf8(selection.model));
  value["manifest"] = selection.manifest.empty()
                          ? gufo::json::Value()
                          : gufo::json::Value(Utf8(selection.manifest));
  value["pointer"] = selection.pointer.empty()
                         ? gufo::json::Value()
                         : gufo::json::Value(Utf8(selection.pointer));
  value["source"] = selection.source;
  return value;
}
bool RecordPreparedDefault(std::string_view local, const fs::path& manifest,
                           std::string& error) {
  try {
    const auto pointer = Pointer(local);
    if (pointer.empty())
      throw std::invalid_argument(
          "LOCALAPPDATA is unavailable; cannot record prepared default");
    const auto path = ManifestPath(manifest);
    if (path.filename() != "bubo-package.json" || !fs::is_regular_file(path))
      throw std::invalid_argument("published prepared manifest is missing");
    auto value = gufo::json::Value::object();
    value["schema_version"] = 1;
    value["manifest_path"] = Utf8(path);
    fs::create_directories(pointer.parent_path());
    prep::PrepareLock lock;
    if (!lock.Acquire(pointer.parent_path() / "prepared-default.lock", error))
      return false;
    return prep::AtomicWrite(pointer, value.dump() + "\n", error);
  } catch (const std::exception& failure) {
    error = failure.what();
    return false;
  }
}
}  // namespace bubo::cli
