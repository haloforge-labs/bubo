#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <thread>

#include "src/cli/prepare/prepare.hpp"

#ifdef _WIN32
#include <windows.h>
#else
#include <poll.h>
#include <unistd.h>
#endif

#include "src/cli/machine_output.hpp"
#include "src/core/crypto/sha256.hpp"
#include "src/core/prepare/metadata_json.hpp"

namespace bubo::prepare {
namespace {
using gufo::json::Value;
// Poll the existing cancellation callbacks. No worker thread needs to join
// at completion. Partial pipe writes and EOF remain observable.
class Input {
public:
  explicit Input(std::istream& input)
      : input_(input), native_(&input == &std::cin) {}
  int Next(std::string& line) {
    if (!native_) {
      if (!std::getline(input_, line))
        return -1;
      if (input_.eof() || line.size() > 65536)
        throw std::invalid_argument("incomplete or oversized input record");
      return 1;
    }
    while (true) {
      if (auto end = buffer_.find('\n'); end != buffer_.npos) {
        if (end > 65536)
          throw std::invalid_argument("input record exceeds 64 KiB");
        line = buffer_.substr(0, end);
        buffer_.erase(0, end + 1);
        return 1;
      }
      if (buffer_.size() > 65536)
        throw std::invalid_argument("input record exceeds 64 KiB");
      char bytes[4096];
#ifdef _WIN32
      const auto handle = GetStdHandle(STD_INPUT_HANDLE);
      DWORD available = 0;
      if (GetFileType(handle) == FILE_TYPE_PIPE) {
        if (!PeekNamedPipe(handle, nullptr, 0, nullptr, &available, nullptr))
          return End();
        if (!available)
          return 0;
      } else if (GetFileType(handle) == FILE_TYPE_CHAR) {
        DWORD pending = 0, read = 0;
        INPUT_RECORD events[128];
        if (!GetNumberOfConsoleInputEvents(handle, &pending))
          return -1;
        if (!pending)
          return 0;
        if (!ReadConsoleInputW(handle, events, std::min<DWORD>(pending, 128),
                               &read))
          return -1;
        for (DWORD i = 0; i < read; ++i) {
          if (events[i].EventType != KEY_EVENT ||
              !events[i].Event.KeyEvent.bKeyDown)
            continue;
          const auto character = events[i].Event.KeyEvent.uChar.UnicodeChar;
          if (character == 26)
            return -1;
          if (character == '\b') {
            if (!buffer_.empty())
              buffer_.pop_back();
          } else if (character == '\r')
            buffer_ += '\n';
          else if (character >= 32 && character < 127)
            buffer_ += static_cast<char>(character);
        }
        continue;
      }
      DWORD count = 0;
      if (!ReadFile(handle, bytes, sizeof(bytes), &count, nullptr) || !count)
        return End();
#else
      pollfd descriptor{STDIN_FILENO, POLLIN, 0};
      if (poll(&descriptor, 1, 0) <= 0)
        return 0;
      const auto count = read(STDIN_FILENO, bytes, sizeof(bytes));
      if (count <= 0)
        return End();
#endif
      buffer_.append(bytes, count);
    }
  }

private:
  int End() const {
    if (!buffer_.empty())
      throw std::invalid_argument("incomplete input record");
    return -1;
  }
  std::istream& input_;
  bool native_;
  std::string buffer_;
};
bool Action(const Value& value, std::string_view action,
            std::string_view operation, std::string_view digest = {}) {
  metadata::Keys(value, {"action", "operation_id", "plan_digest"});
  if (const auto* supplied = value.find("plan_digest"))
    if (!supplied->is_string() || supplied->str().size() != 64 ||
        supplied->str().find_first_not_of("0123456789abcdef") !=
            std::string::npos)
      throw std::invalid_argument("invalid plan digest");
  return value.member_str("action") == action &&
         value.member_str("operation_id") == operation &&
         (digest.empty() || value.member_str("plan_digest") == digest);
}
Value ParseInput(std::string_view line) {
  try {
    return metadata::Parse(line);
  } catch (const std::exception&) {
    throw std::invalid_argument("invalid input JSON");
  }
}
bool WritePlan(bubo::cli::MachineWriter& writer, const Value& plan) {
  // Emit component records before approval. Bound arrays as well as legal
  // text; every part binds to the complete, unsplit plan digest.
  std::vector<Value> records;
  auto header = plan;
  header["sources"] = Value::array();
  header["licenses"] = Value::array();
  header["paths"] = Value::object();
  header["volumes"] = Value::array();
  header["record_kind"] = "plan";
  records.push_back(header);
  const auto part = [&](const char* kind) {
    auto row = header;
    row["record_kind"] = kind;
    return row;
  };
  for (const auto& source : metadata::Required(plan, "sources").items()) {
    auto row = part("source");
    row["sources"].push_back(source);
    records.push_back(std::move(row));
  }
  for (const auto& [key, path] : metadata::Required(plan, "paths").members()) {
    auto row = part("path");
    row["paths"][key] = path;
    records.push_back(std::move(row));
  }
  for (const auto& volume : metadata::Required(plan, "volumes").items()) {
    auto row = part("volume");
    row["volumes"].push_back(volume);
    records.push_back(std::move(row));
  }
  for (const auto& license : metadata::Required(plan, "licenses").items()) {
    const auto& text = metadata::String(metadata::Required(license, "text"));
    for (std::size_t offset = 0; offset < text.size();) {
      auto count = std::min<std::size_t>(64 * 1024, text.size() - offset);
      // A UTF-8 continuation byte must not begin the next record.
      while (offset + count < text.size() && count &&
             (static_cast<unsigned char>(text[offset + count]) & 0xc0) == 0x80)
        --count;
      auto legal = license;
      legal["text"] = text.substr(offset, count);
      legal["text_offset"] = offset;
      legal["text_bytes"] = text.size();
      auto row = part("license");
      row["licenses"].push_back(legal);
      records.push_back(row);
      offset += count;
    }
  }
  for (std::size_t i = 0; i < records.size(); ++i) {
    records[i]["part"] = i;
    records[i]["parts"] = records.size();
    if (!writer.Write("plan", records[i]))
      return false;
  }
  return true;
}
}  // namespace

namespace {
// A volume GUID path (\\?\Volume{...}\) shows as its drive letter, "C:".
std::string VolumeDisplayName(const std::string& name) {
#ifdef _WIN32
  if (name.starts_with("\\\\?\\Volume{")) {
    std::wstring volume(name.begin(), name.end());  // The GUID path is ASCII.
    if (volume.back() != L'\\')
      volume += L'\\';
    wchar_t paths[1024] = {};
    DWORD length = 0;
    if (GetVolumePathNamesForVolumeNameW(volume.c_str(), paths, 1024,
                                         &length) &&
        paths[0]) {
      std::wstring first(paths);
      if (first.size() == 3 && first[1] == L':' && first[2] == L'\\')
        return {static_cast<char>(first[0]), ':'};
      const auto bytes = WideCharToMultiByte(CP_UTF8, 0, first.c_str(), -1,
                                             nullptr, 0, nullptr, nullptr);
      if (bytes > 1) {
        std::string utf8(static_cast<std::size_t>(bytes - 1), '\0');
        WideCharToMultiByte(CP_UTF8, 0, first.c_str(), -1, utf8.data(), bytes,
                            nullptr, nullptr);
        return utf8;
      }
    }
  }
#endif
  return name;
}
// One decimal. Needed space rounds up and free space rounds down, so a
// shortage never prints as two equal numbers.
std::string Gib(std::uint64_t bytes, bool up) {
  const auto tenths = static_cast<double>(bytes) * 10.0 / (1ULL << 30);
  const auto rounded = up ? std::ceil(tenths) : std::floor(tenths);
  const auto whole = static_cast<std::uint64_t>(rounded);
  return std::to_string(whole / 10) + '.' + std::to_string(whole % 10) + " GiB";
}
}  // namespace

std::string NotEnoughSpaceText(const ConsentVolume& volume,
                               std::string_view option) {
  return "Not enough free space on " + VolumeDisplayName(volume.name) +
         ". Needs " + Gib(volume.required_bytes, true) + "; " +
         Gib(volume.free_bytes, false) +
         " free. Free some space, or choose another drive with " +
         std::string(option) + ".";
}

Value PlanVolume(const ConsentVolume& volume) {
  constexpr std::uint64_t margin = 2ULL * 1024 * 1024 * 1024;
  const auto safety = std::min(margin, volume.required_bytes);
  const auto writes = volume.required_bytes - safety;
  auto row = Value::object();
  row["name"] = volume.name;
  row["free_bytes"] = volume.free_bytes;
  row["required_bytes"] = volume.required_bytes;
  row["safety_margin_bytes"] = safety;
  row["bytes_to_write"] = writes;
  row["free_after_download_bytes"] =
      volume.free_bytes - std::min(volume.free_bytes, writes);
  return row;
}

std::string PlanDigest(const Value& value) {
  // Observed free space can change without changing the approved action.
  // Source identities, required space, paths, states and legal text bind it.
  auto fixed = Value::object();
  for (const auto& [key, field] : value.members()) {
    if (key == "plan_digest")
      continue;
    if (key == "volumes") {
      auto volumes = Value::array();
      for (const auto& volume : field.items()) {
        auto row = Value::object();
        row["name"] = metadata::Required(volume, "name");
        row["required_bytes"] = metadata::Required(volume, "required_bytes");
        volumes.push_back(row);
      }
      fixed[key] = volumes;
    } else
      fixed[key] = field;
  }
  const auto text = fixed.dump();
  return gufo::crypto::Sha256Hex(std::span(
      reinterpret_cast<const std::uint8_t*>(text.data()), text.size()));
}
Value PlanSource(std::string_view id, std::string_view url, std::uint64_t size,
                 std::string_view sha, std::string_view repo,
                 std::string_view revision, std::string_view path,
                 std::string_view license) {
  auto value = Value::object();
  value["file_id"] = std::string(id);
  value["url"] = std::string(url);
  value["size"] = size;
  value["sha256"] = std::string(sha);
  value["repository"] = std::string(repo);
  value["revision"] = std::string(revision);
  value["path"] = std::string(path);
  value["license_id"] = std::string(license);
  return value;
}
void ReportProgress(const Options& options, std::string_view component,
                    std::string_view phase, std::string_view file,
                    std::uint64_t done, std::uint64_t total,
                    std::uint64_t resume, std::string_view method) {
  if (!options.progress)
    return;
  auto value = Value::object();
  value["component"] = std::string(component);
  value["phase"] = std::string(phase);
  value["file_id"] = std::string(file);
  value["bytes_done"] = done;
  value["bytes_total"] = total ? Value(total) : Value();
  value["resume_offset"] = resume;
  value["verification_method"] = std::string(method);
  options.progress(std::move(value));
}

int RunMachine(const Options& requested, const PreparePhase& model,
               const PreparePhase& runtime, std::istream& stream,
               bubo::cli::MachineWriter& writer, std::ostream& err,
               const std::function<bool()>& signal_cancelled,
               const std::function<int(std::string&)>& read_line) {
  Input input(stream);
  const auto next = [&](std::string& line) {
    const auto state = read_line ? read_line(line) : input.Next(line);
    if (state > 0 && line.size() > 65536)
      throw std::invalid_argument("input record exceeds 64 KiB");
    return state;
  };
  bool approved = false, cancelled = false, changed = false, broken = false;
  bool invalid_input = false;
  bool preflight_work = false;
  bool approval_waiting = false;
  Options options = requested;
  // A download reports once per network chunk. Write at most one record per
  // 250 ms for each file and phase, plus the first one and every record with
  // bytes_done == bytes_total.
  std::map<std::string, std::chrono::steady_clock::time_point> progress_sent;
  std::mutex progress_mutex;
  options.progress = [&](Value value) {
    if (!approved)
      preflight_work = true;
    {
      const auto* total = value.find("bytes_total");
      const bool last = total && !total->is_null() &&
                        value.member_size("bytes_done") ==
                            value.member_size("bytes_total");
      const auto key = value.member_str("component") + '\n' +
                       value.member_str("phase") + '\n' +
                       value.member_str("file_id");
      const auto now = std::chrono::steady_clock::now();
      std::lock_guard lock(progress_mutex);
      const auto found = progress_sent.find(key);
      if (!last && found != progress_sent.end() &&
          now - found->second < std::chrono::milliseconds(250))
        return;
      progress_sent[key] = now;
    }
    if (!writer.Write("progress", std::move(value)))
      broken = true;
  };
  const auto poll = [&] {
    if (!writer.Connected())
      broken = true;
    if (signal_cancelled && signal_cancelled())
      cancelled = true;
    try {
      // Native stdin can carry cancel while preflight hashes existing files.
      // Approval replies are read only after approval_required. Injected
      // readers retain their request-driven test/embedding interface.
      if ((approved ||
           (!approval_waiting && !read_line && &stream == &std::cin)) &&
          !cancelled) {
        std::string line;
        for (int state = next(line); state != 0; state = next(line)) {
          if (state < 0) {
            // A read-only, prepared no-op can succeed with closed stdin.
            // A non-ready plan will decline at the approval boundary.
            cancelled = approved || preflight_work;
            break;
          }
          if (Action(ParseInput(line), "cancel", writer.OperationId())) {
            cancelled = true;
            break;
          }
          throw std::invalid_argument("expected cancel for this operation");
        }
      }
    } catch (const std::exception&) {
      // Preparation callbacks map their own failures. Keep input errors
      // distinct even when a callback catches exceptions during verification.
      invalid_input = true;
      cancelled = true;
    }
    return cancelled || broken;
  };
  options.cancelled = poll;
  if (!writer.Write("capabilities", bubo::cli::Capabilities()))
    return 3;
  // Phases write their failure as "prepare: E<code> <text>" to stderr. The
  // error record repeats that text as `detail`, so a consumer that does not
  // keep stderr still has the cause.
  bubo::cli::LastLineCapture capture(err.rdbuf(), "prepare: E");
  std::ostream errors(&capture);
  const auto fail = [&](int code, std::string_view stage,
                        std::string detail = {}) {
    if (invalid_input) {
      code = 2;
      stage = "input";
    } else if (cancelled || broken)
      code = 3;
    if (!detail.empty())
      errors << "prepare: E" << code << ' ' << detail << '\n';
    else {
      // Only a line for this code is the cause; an earlier line is not.
      const auto prefix = "prepare: E" + std::to_string(code) + ' ';
      const auto last = capture.Last();
      if (last.starts_with(prefix))
        detail = last.substr(prefix.size());
    }
    errors.flush();
    writer.Write("error", bubo::cli::PrepareError(code, stage, detail));
    return code;
  };
  try {
    for (;;) {
      approved = false;
      changed = false;
      preflight_work = false;
      std::map<std::string, PreparationPlan> plans;
      const auto preview = [&](const char* component,
                               const PreparePhase& phase) {
        auto event = Value::object();
        event["component"] = component;
        event["phase"] = "preflight";
        if (!writer.Write("preflight", event))
          return 3;
        auto& plan = plans[component];
        const auto code = phase(options, stream, errors, errors, &plan, false);
        if (!code) {
          plan.machine["plan_digest"] = PlanDigest(plan.machine);
          if (!WritePlan(writer, plan.machine))
            return 3;
        }
        return code;
      };
      if (options.step != Step::Runtime)
        if (const auto code = preview("model", model))
          return fail(code, "preflight");
      if (options.step != Step::Model)
        if (const auto code = preview("runtime", runtime))
          return fail(code, "preflight");
      if (options.plan_only) {
        // Plan only: every plan part is written. No approval, no work.
        bool plan_ready = true;
        for (const auto& [component, plan] : plans)
          plan_ready &= plan.ready;
        auto result = Value::object();
        result["exit_code"] = 0;
        result["prepared"] = plan_ready;
        result["manifest_path"] = Value();
        result["plan_only"] = true;
        return writer.Write("result", result) ? 0 : 3;
      }
      std::map<std::string, ConsentVolume> volumes;
      for (const auto& [component, plan] : plans)
        for (const auto& volume : plan.volumes) {
          auto [row, inserted] = volumes.try_emplace(volume.name, volume);
          if (!inserted) {
            row->second.free_bytes =
                std::min(row->second.free_bytes, volume.free_bytes);
            constexpr std::uint64_t margin = 2ULL * 1024 * 1024 * 1024;
            const auto extra =
                volume.required_bytes - std::min(margin, volume.required_bytes);
            if (extra > std::numeric_limits<std::uint64_t>::max() -
                            row->second.required_bytes)
              return fail(7, "preflight",
                          "combined space calculation overflow");
            row->second.required_bytes += extra;
          }
        }
      for (const auto& [name, volume] : volumes)
        if (volume.free_bytes < volume.required_bytes)
          return fail(7, "preflight", NotEnoughSpaceText(volume));
      bool ready = true;
      auto binding = Value::object();
      auto components = Value::array();
      for (const auto& [component, plan] : plans) {
        ready &= plan.ready;
        binding[component] = plan.machine.member_str("plan_digest");
        components.push_back(component);
      }
      if (!ready) {
        const auto digest = PlanDigest(binding);
        if (!options.yes) {
          auto request = Value::object();
          request["plan_digest"] = digest;
          request["components"] = components;
          approval_waiting = true;
          if (!writer.Write("approval_required", request))
            return 3;
          for (;;) {
            if (poll())
              return fail(3, "approval");
            std::string line;
            const auto state = next(line);
            if (state < 0)
              return fail(3, "approval");
            if (!state) {
              std::this_thread::sleep_for(std::chrono::milliseconds(20));
              continue;
            }
            const auto value = ParseInput(line);
            if (Action(value, "cancel", writer.OperationId()))
              return fail(3, "approval");
            if (Action(value, "decline", writer.OperationId(), digest))
              return fail(3, "approval");
            if (!Action(value, "approve", writer.OperationId(), digest))
              return fail(2, "approval",
                          "the reply must approve this operation_id and "
                          "plan_digest");
            break;
          }
          approval_waiting = false;
        }
        approved = true;
        options.authorize_plan = [&](const Value& value) {
          const auto found = plans.find(value.member_str("component"));
          changed = found == plans.end() ||
                    PlanDigest(value) !=
                        found->second.machine.member_str("plan_digest");
          return !changed;
        };
        for (const char* component : {"runtime", "model"}) {
          if (!plans.contains(component) || plans.at(component).ready)
            continue;
          const auto& phase =
              std::string_view(component) == "model" ? model : runtime;
          if (poll())
            return fail(3, "work");
          const auto code =
              phase(options, stream, errors, errors, nullptr, true);
          if (changed)
            break;
          if (code)
            return fail(code, "work");
        }
        if (changed)
          continue;
        if (poll())
          return fail(3, "work");
      }
      std::string runtime_dir;
      for (const auto& [component, previous] : plans) {
        if (poll())
          return fail(3, "completion");
        PreparationPlan final;
        const auto& phase = component == "model" ? model : runtime;
        if (const auto code =
                phase(options, stream, errors, errors, &final, false))
          return fail(code, "completion");
        if (!final.ready)
          return fail(7, "completion");
        if (component == "runtime")
          runtime_dir = final.runtime_dir;
      }
      if (poll())
        return fail(3, "completion");
      auto result = Value::object();
      std::string manifest;
      if (options.step != Step::Runtime) {
        const auto path = std::filesystem::absolute(
                              std::filesystem::u8path(ModelRoot(options)) /
                              "bubo-package.json")
                              .lexically_normal()
                              .u8string();
        manifest = std::string(path.begin(), path.end());
      }
      if (options.step != Step::Runtime && options.publish_default)
        if (const auto code = options.publish_default(manifest))
          return fail(code, "prepared_default");
      result["exit_code"] = 0;
      result["prepared"] = true;
      result["manifest_path"] =
          options.step == Step::Runtime ? Value() : Value(manifest);
      // A runtime in a chosen folder is found only through BUBO_RUNTIME_DIR.
      if (!options.runtime_dir.empty() && !runtime_dir.empty())
        result["runtime_environment"] = "BUBO_RUNTIME_DIR=" + runtime_dir;
      return writer.Write("result", result) ? 0 : 3;
    }
  } catch (const std::invalid_argument& error) {
    return fail(2, "input", error.what());
  } catch (const std::exception& error) {
    return fail(9, "machine_output", error.what());
  }
}
}  // namespace bubo::prepare
