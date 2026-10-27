#include "src/cli/models_support.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <fstream>
#include <functional>
#include <iostream>
#include <regex>
#include <set>
#include <sstream>

#include "src/cli/arg_parser.hpp"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

#include "src/cli/machine_adapter.hpp"
#include "src/cli/machine_exit.hpp"
#include "src/cli/machine_output.hpp"
#include "src/cli/memory_advice.hpp"
#include "src/cli/prepare/prepare.hpp"
#include "src/cli/serve/memory_estimate.hpp"
#include "src/cli/serve/memory_policy.hpp"
#include "src/cli/settings_support.hpp"
#include "src/core/prepare/metadata_json.hpp"
#include "src/core/session_state_bytes.hpp"
#include "src/models/gemma4/config.hpp"
#include "src/models/qwen/hip/execution_policy.hpp"
#include "src/models/qwen/state.hpp"
#include "src/models/qwen36_a3b/config.hpp"
#include "src/models/qwen38_flash_next/config.hpp"

namespace bubo::cli {
namespace fs = std::filesystem;
using gufo::json::Value;
namespace core = gufo::core;

namespace {
std::string Utf8(const fs::path& path) {
  const auto text = path.u8string();
  return {text.begin(), text.end()};
}
std::string Lower(std::string text) {
  std::transform(text.begin(), text.end(), text.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return text;
}
// A link the scan must not follow. The reparse tag tells a link from a
// OneDrive placeholder, which is an ordinary folder or file.
bool IsReparsePoint(const fs::path& path) {
#ifdef _WIN32
  WIN32_FIND_DATAW data{};
  const HANDLE found = FindFirstFileW(path.c_str(), &data);
  if (found == INVALID_HANDLE_VALUE)
    return false;
  FindClose(found);
  return (data.dwFileAttributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0 &&
         IsLinkReparseTag(data.dwReserved0);  // the tag, for a reparse point
#else
  std::error_code ec;
  return fs::is_symlink(path, ec);
#endif
}
Value OptionalString(const std::string& text) {
  return text.empty() ? Value() : Value(text);
}
std::uint64_t CeilMiB(std::uint64_t bytes) {
  return (bytes + 1048575ULL) / 1048576ULL;
}
}  // namespace

bool IsLinkReparseTag(std::uint32_t tag) {
  constexpr std::uint32_t kSymlink = 0xA000000CU;  // IO_REPARSE_TAG_SYMLINK
  constexpr std::uint32_t kMountPoint =
      0xA0000003U;  // IO_REPARSE_TAG_MOUNT_POINT
  return tag == kSymlink || tag == kMountPoint;
}

std::optional<std::uint64_t> HostAvailableBytes() {
  return ReadPlanningMemoryBudgets().host_available;
}

bool IsFilesystemRoot(const fs::path& input) {
  if (input.empty())
    return false;
  const auto path = input.lexically_normal();
  if (path.has_root_name() && !path.has_root_directory() &&
      path.relative_path().empty())
    return true;  // "C:"
  if (path.has_root_path() && path.relative_path().empty())
    return true;  // "C:\" or "/"
  const auto name = path.root_name().string();
  if (name.starts_with("\\\\")) {  // network share root: \\server\share
    std::size_t parts = 0;
    for (const auto& part : path.relative_path())
      if (!part.empty() && part != ".")
        ++parts;
    return parts <= 1;
  }
  return false;
}

// ---- header inspection -----------------------------------------------------

ArchitectureSupport CheckArchitectureSupport(const core::GgufReader& reader) {
  ArchitectureSupport support;
  const auto architecture = reader.GetMetadataString("general.architecture");
  if (!architecture) {
    support.reason = "The file has no general.architecture key.";
    return support;
  }
  const std::string arch(*architecture);
  if (arch == "clip") {
    support.sidecar = true;
    support.reason =
        "This file is a vision projector. Use it as a side file of a model.";
    return support;
  }
  if (arch == "deepseek4") {
    support.runnable = true;
    support.route = "deepseek_v4_flash";
    return support;
  }
  // Serve sends gemma4 files to the Gemma 4 runner. Its loader starts with
  // the same config check (ModelWeights::Bind -> Config::FromGguf); the
  // tensors are checked when serve loads the file.
  if (arch == "gemma4") {
    std::string error;
    if (gufo::models::gemma4::Config::FromGguf(reader, &error)) {
      support.runnable = true;
      support.route = "gemma4";
    } else
      support.reason = "The engine cannot load this gemma4 file: " + error;
    return support;
  }
  if (arch == "gemma4-assistant") {
    support.sidecar = true;
    support.reason =
        "This file is the draft (MTP) side file of a Gemma 4 model.";
    return support;
  }
  if (arch == "diffusion_gemma") {
    support.reason =
        "DiffusionGemma runs from its safetensors folder. Serve does not load "
        "it from a GGUF file.";
    return support;
  }
  if (arch == "qwen4exp") {
    std::string error;
    const auto trunk =
        gufo::models::qwen38_flash_next::Config::FromGguf(reader, true, &error);
    if (trunk) {
      support.runnable = true;
      support.route = "qwen38_flash_next";
      return support;
    }
    std::string draft_error;
    if (gufo::models::qwen38_flash_next::Config::FromGguf(reader, false,
                                                          &draft_error) &&
        reader.GetMetadataUint32("qwen4exp.nextn_predict_layers").value_or(0) >
            0) {
      support.sidecar = true;
      support.reason =
          "This file is the draft (MTP) side file of a Flash Next model.";
      return support;
    }
    support.reason = "The engine cannot load this qwen4exp file: " + error;
    return support;
  }
  if (arch == gufo::models::qwen36_a3b::kArch) {
#ifdef GUFO_WITH_A3B
    gufo::models::qwen36_a3b::Config config;
    std::string error;
    if (gufo::models::qwen36_a3b::ReadConfig(reader, config, &error)) {
      support.runnable = true;
      support.route = "qwen36_a3b";
    } else
      support.reason = "The engine cannot load this qwen35moe file: " + error;
#else
    support.reason =
        "This build was made without Qwen3.6-35B-A3B (qwen35moe) support.";
#endif
    return support;
  }
  // Everything else goes to the dense Qwen loader, as in the server. The
  // loader reads the same header keys and tensor table the server reads.
  std::string error;
  if (gufo::models::QwenModelWeights::LoadFromGguf(reader, &error)) {
    support.runnable = true;
    support.route = "qwen";
    return support;
  }
  support.reason = "The engine in this build does not support the '" + arch +
                   "' architecture. The Qwen loader said: " + error;
  return support;
}

Inspection InspectGguf(const fs::path& path) {
  Inspection result;
  auto& facts = result.facts;
  facts.path = Utf8(path);
  facts.file_name = Utf8(path.filename());
  std::error_code ec;
  facts.size_bytes = fs::file_size(path, ec);
  if (ec) {
    facts.error = "Cannot read the file size.";
    result.support.reason = facts.error;
    return result;
  }
  facts.total_bytes = facts.size_bytes;
  std::string error;
  auto reader = core::GgufReader::OpenHeaderFile(path, &error);
  if (!reader) {
    facts.error = error.empty() ? "Not a readable GGUF file." : error;
    result.support.reason = "The header cannot be read: " + facts.error;
    return result;
  }
  facts.header_ok = true;
  if (const auto name = reader->GetMetadataString("general.name"))
    facts.model_name = std::string(*name);
  if (const auto arch = reader->GetMetadataString("general.architecture")) {
    facts.architecture = std::string(*arch);
    auto context =
        reader->GetMetadataUint64(facts.architecture + ".context_length");
    if (!context)
      if (const auto small =
              reader->GetMetadataUint32(facts.architecture + ".context_length"))
        context = *small;
    facts.native_context = context;
  }
  const auto count = reader->GetMetadataUint32("split.count").value_or(1);
  const auto number = reader->GetMetadataUint32("split.no").value_or(0);
  if (count > 1) {
    facts.shards_expected = count;
    facts.split_follower = number != 0;
    static const std::regex pattern(R"((.*)-(\d{5})-of-(\d{5})\.gguf$)",
                                    std::regex::icase);
    std::smatch match;
    const auto name = facts.file_name;
    if (std::regex_match(name, match, pattern)) {
      std::uint64_t total = 0;
      unsigned found = 0;
      for (unsigned i = 1; i <= count; ++i) {
        char index[16], of[16];
        std::snprintf(index, sizeof(index), "%05u", i);
        std::snprintf(of, sizeof(of), "%05u", count);
        const auto sibling =
            path.parent_path() / fs::u8path(match[1].str() + "-" + index +
                                            "-of" + "-" + of + ".gguf");
        std::error_code sibling_ec;
        const auto size = fs::file_size(sibling, sibling_ec);
        if (!sibling_ec) {
          total += size;
          ++found;
        }
      }
      facts.total_bytes = total;
      facts.shards_found = found;
    }
  }
  result.support = CheckArchitectureSupport(*reader);
  result.reader = std::move(reader);
  return result;
}

// ---- scanner ---------------------------------------------------------------

ScanResult ScanModelFolders(const ScanOptions& options) {
  ScanResult result;
  std::set<std::string> visited;
  std::set<std::string> seen_files;
  std::size_t folder_count = 0;
  for (const auto& folder : options.folders) {
    const auto name = Utf8(folder);
    const auto refuse = [&](const char* reason) {
      result.warnings.push_back({name, reason});
    };
    if (folder.empty()) {
      refuse("The folder name is empty.");
      continue;
    }
    if (IsFilesystemRoot(folder)) {
      refuse("A drive or share root is never scanned. Choose a model folder.");
      continue;
    }
    std::error_code ec;
    if (!fs::is_directory(folder, ec)) {
      refuse("The folder does not exist.");
      continue;
    }
    if (IsReparsePoint(folder)) {
      refuse("The folder is a link or junction. Links are not followed.");
      continue;
    }
    result.scanned_folders.push_back(name);
    struct Item {
      fs::path path;
      int depth;
    };
    std::vector<Item> stack{{folder, 0}};
    bool depth_skipped = false, file_limit = false, folder_limit = false;
    while (!stack.empty()) {
      const auto [path, depth] = stack.back();
      stack.pop_back();
      std::error_code canonical_ec;
      auto key = Lower(Utf8(fs::weakly_canonical(path, canonical_ec)));
      if (canonical_ec)
        key = Lower(Utf8(path));
      if (!visited.insert(key).second)
        continue;  // loop or a folder named twice
      if (++folder_count > options.max_folders) {
        folder_limit = true;
        break;
      }
      std::error_code marker_ec;
      if (fs::is_regular_file(path / "bubo-package.json", marker_ec)) {
        result.packages.push_back(path);  // one package: do not enter it
        continue;
      }
      std::error_code iterator_ec;
      fs::directory_iterator iterator(
          path, fs::directory_options::skip_permission_denied, iterator_ec);
      if (iterator_ec) {
        result.warnings.push_back({Utf8(path), "The folder cannot be read."});
        continue;
      }
      for (const fs::directory_iterator end; iterator != end;
           iterator.increment(iterator_ec)) {
        if (iterator_ec)
          break;
        const auto& entry = *iterator;
        const auto entry_path = entry.path();
        if (IsReparsePoint(entry_path)) {
          result.warnings.push_back(
              {Utf8(entry_path), "A link or junction is not followed."});
          continue;
        }
        std::error_code type_ec;
        if (entry.is_directory(type_ec)) {
          if (depth + 1 > options.max_depth)
            depth_skipped = true;
          else
            stack.push_back({entry_path, depth + 1});
          continue;
        }
        if (!entry.is_regular_file(type_ec))
          continue;
        const auto extension = Lower(Utf8(entry_path.extension()));
        if (extension != ".gguf")
          continue;
        if (!seen_files.insert(Lower(Utf8(entry_path))).second)
          continue;
        if (result.files.size() >= options.max_files) {
          file_limit = true;
          break;
        }
        result.files.push_back(entry_path);
      }
      if (file_limit)
        break;
    }
    if (depth_skipped)
      result.warnings.push_back(
          {name, "Folders deeper than " + std::to_string(options.max_depth) +
                     " levels below this folder were not scanned."});
    if (file_limit)
      result.warnings.push_back({name, "The scan stopped after " +
                                           std::to_string(options.max_files) +
                                           " files."});
    if (folder_limit)
      result.warnings.push_back({name, "The scan stopped after " +
                                           std::to_string(options.max_folders) +
                                           " folders."});
  }
  std::sort(result.files.begin(), result.files.end());
  std::sort(result.packages.begin(), result.packages.end());
  return result;
}

// ---- estimate --------------------------------------------------------------

namespace {
EstimateResult NoneYet(std::string reason, EstimateResult base = {}) {
  base.state = "none_yet";
  base.reason = std::move(reason);
  base.memory_estimate = Value();
  return base;
}

// A coefficient table built from header facts. The strings the table views
// live in this object.
struct HeaderTable {
  std::string package_id;
  MemoryModelTable table;
};

std::unique_ptr<HeaderTable> MakeHeaderTable(
    const std::string& id, std::uint64_t weights, std::uint64_t context_max,
    std::uint64_t max_sessions,
    const std::function<std::uint64_t(std::uint64_t)>& state_bytes) {
  auto owned = std::make_unique<HeaderTable>();
  owned->package_id = id;
  auto& t = owned->table;
  t.package_id = owned->package_id;
  t.version = 1;
  t.reference_context = context_max;
  const auto fixed = state_bytes(0);
  const auto at_reference = state_bytes(context_max);
  t.fixed_mib = CeilMiB(fixed);
  t.session_mib = std::max(CeilMiB(at_reference), t.fixed_mib + 1);
  t.main_weights_bytes = weights;
  t.scratch_min_bytes = 0;
  t.scratch_max_bytes = 0;
  t.reserve_bytes = MemoryCoefficientsV1::reserve_bytes;
  t.estimate_context_limit = context_max;
  t.context_max = context_max;
  t.context_min = std::min<std::uint64_t>(4096, context_max);
  t.max_sessions = max_sessions;
  t.cache_on_host = false;
  t.reference_source_path = "src/core/session_state_bytes.hpp";
  return owned;
}

void FinishHeaderEstimate(
    Value& estimate, const HeaderTable& owned,
    const std::function<std::uint64_t(std::uint64_t)>& state_bytes) {
  estimate["evidence"] = "estimated_from_header";
  estimate["reason"] =
      "Estimated from the file header. Scratch memory while the model reads "
      "a long prompt and the RAM cache are not included, so the real need is "
      "higher. Fit is not verified.";
  estimate["formula"] =
      "weights + session_state_total (lower bound; scratch and RAM "
      "cache not included)";
  estimate["scratch_phase_allowance"]["min"] = MemoryAmount(0, "unknown");
  estimate["scratch_phase_allowance"]["max"] = MemoryAmount(0, "unknown");
  auto reference =
      MemoryAmount(state_bytes(owned.table.reference_context), "estimated");
  reference["context_tokens"] = owned.table.reference_context;
  reference["conditions"] =
      "State size of one session at the model's own context length, from the "
      "header keys and the engine's state-size functions.";
  reference["source_path"] = std::string(owned.table.reference_source_path);
  estimate["reference_coefficient"] = std::move(reference);
}

void FillFitInputs(EstimateResult& result, std::uint64_t weights,
                   std::optional<std::uint64_t> native_context,
                   std::uint64_t context, std::uint64_t sessions,
                   std::optional<std::uint64_t> available) {
  auto& inputs = result.fit_inputs;
  inputs["weights_bytes"] = weights;
  inputs["native_context_tokens"] =
      native_context ? Value(*native_context) : Value();
  inputs["context_tokens"] = context;
  inputs["sessions"] = sessions;
  inputs["available_bytes"] = available ? Value(*available) : Value();
}

EstimateResult RunTable(ReleaseMemoryEstimate planning, EstimateResult result) {
  if (planning.value.is_null())
    return NoneYet(planning.reason.empty() ? "The estimate is not available."
                                           : planning.reason,
                   std::move(result));
  result.state = "estimated";
  result.reason.clear();
  result.memory_estimate = std::move(planning.value);
  return result;
}

// Device bytes of a bound vision projector on the dense Qwen and Qwen3.6-A3B
// runners. qwen::vision::Encoder::Upload (src/models/qwen/vision/encoder.hip)
// puts every projector tensor except the two F32 patch-embedding halves on
// the device at 4 bytes (F32) or 2 bytes (other types) per element, plus the
// fused BF16 patch weight (1152 x 1536 values). It uploads on the first
// image; these runners reserve nothing for vision at startup.
std::optional<std::uint64_t> QwenVisionUploadBytes(const fs::path& path,
                                                   std::string* error) {
  const auto reader = core::GgufReader::OpenHeaderFile(path, error);
  if (!reader)
    return std::nullopt;
  std::uint64_t bytes = 1152ULL * 1536 * 2;
  for (const auto& tensor : reader->GetTensors()) {
    if (tensor.name == "v.patch_embd.weight" ||
        tensor.name == "v.patch_embd.weight.1")
      continue;
    bytes +=
        tensor.ElementCount() * (tensor.type == core::GgmlType::kF32 ? 4 : 2);
  }
  return bytes;
}

constexpr const char* kRamCacheNote =
    "RAM cache: load-dependent. In this build only the Flash Next runner sizes "
    "its cache from the totals. This runner sizes it from free memory when the "
    "server starts, so the cache is not in this estimate.";

gufo::server::ReleaseResolution EstimateReleaseSettings(
    const fs::path& path, const EstimateRequest& request, bool manifest = false,
    std::string* package_id = nullptr) {
  std::vector<std::string> args{manifest ? "--manifest" : "--model",
                                Utf8(path),
                                "--profile",
                                request.profile,
                                "--sessions",
                                std::to_string(request.sessions)};
  if (request.context) {
    args.push_back("--context");
    args.push_back(std::to_string(request.context));
  }
  std::vector<const char*> raw;
  for (const auto& arg : args)
    raw.push_back(arg.c_str());
  // Same manifest/header/profile resolution as serve, without environment
  // injection or disk-cache identity writes. No model payload is loaded.
  auto startup = gufo::server::PrepareReleaseSettings(raw, {}, true);
  if (!startup.error.empty())
    throw std::invalid_argument(startup.error);
  if (package_id)
    *package_id =
        startup.package.manifest_path.empty() ? "" : startup.package.package_id;
  return std::move(startup.resolved);
}
}  // namespace

EstimateResult EstimateForGguf(const Inspection& inspection,
                               const EstimateRequest& request) {
  EstimateResult base;
  const auto& facts = inspection.facts;
  if (!facts.header_ok)
    return NoneYet("The header cannot be read: " + facts.error, base);
  if (inspection.support.sidecar)
    return NoneYet(inspection.support.reason, base);
  if (!inspection.support.runnable)
    return NoneYet(inspection.support.reason, base);
  const auto& reader = *inspection.reader;
  const auto& route = inspection.support.route;
  if (route == "gemma4")
    return NoneYet("Bubo has no memory estimate for this model yet.", base);
  const std::uint64_t weights =
      request.weights_bytes ? *request.weights_bytes : facts.total_bytes;
  if (facts.shards_found != facts.shards_expected)
    base.notes.push_back(
        "Only " + std::to_string(facts.shards_found) + " of " +
        std::to_string(facts.shards_expected) +
        " shards are present. The weights term is smaller than "
        "the full model.");
  std::string package_id;
  auto resolved = EstimateReleaseSettings(fs::u8path(facts.path), request,
                                          false, &package_id);

  if (route == "qwen38_flash_next") {
    const auto* lowmemory =
        package_id == prepare::PackageId(prepare::ModelSet::LowMemory) ||
                facts.file_name ==
                    prepare::ModelCatalogue(prepare::ModelSet::LowMemory)[0]
                        .file_name
            ? FindCatalogueEntry(
                  ShippedCatalogue().entries,
                  prepare::PackageId(prepare::ModelSet::LowMemory))
            : nullptr;
    const auto estimate_id =
        lowmemory ? lowmemory->id : std::string(prepare::kPackageId);
    const auto* shipped = MemoryTableFor(estimate_id);
    if (shipped == nullptr)
      return NoneYet("No coefficient table for Flash Next.", base);
    auto table = *shipped;
    // A prepared package uses GPU-resident coefficients, independently of its
    // download sizes, just as serve does. Unpackaged files use their own size.
    if (lowmemory && lowmemory->measured_main_resident_bytes)
      table.main_weights_bytes = *lowmemory->measured_main_resident_bytes;
    else if (!lowmemory && package_id.empty())
      table.main_weights_bytes = weights;
    auto planning = BuildPlanningReleaseMemoryEstimate(
        resolved, estimate_id, request.available_bytes,
        request.host_available_bytes, &table, request.capacity_budgets,
        request.context != 0);
    const auto& inputs = planning.inputs;
    base.evidence_label =
        lowmemory ? "estimated from the calibrated lower-memory coefficient"
                  : "estimated from the file header and the measured Flash "
                    "Next coefficient";
    base.notes.push_back(
        package_id.empty()
            ? "Selected side files use the package coefficients; the main "
              "weights term is the selected file size."
            : "Prepared package weights use GPU-resident coefficients, "
              "independently of download sizes.");
    base.notes.push_back(
        "Cache mode and allowance follow serve's resolved release settings. "
        "Cache ceilings do not allocate storage at startup; on-demand snapshot "
        "growth is excluded from both startup estimates.");
    if (!planning.value.is_null())
      FillFitInputs(base, planning.value.find("weights")->member_size("bytes"),
                    facts.native_context, inputs.context, inputs.sessions,
                    inputs.available_bytes);
    return RunTable(std::move(planning), std::move(base));
  }

  std::function<std::uint64_t(std::uint64_t)> state_bytes;
  std::uint64_t context_max = facts.native_context.value_or(0);
  std::uint64_t max_sessions = 1;
  try {
    if (route == "qwen") {
      std::string error;
      const auto config = reader.ExtractModelConfig(&error);
      if (!config)
        return NoneYet("The Qwen header keys are not readable: " + error, base);
      const auto policy = gufo::hip::QwenExecutionPolicy::Production();
      const std::size_t kv_bytes =
          policy.UsesFp16AttentionKv() ? sizeof(std::uint16_t) : sizeof(float);
      const std::size_t recurrent_bytes =
          gufo::hip::QwenRecurrentStateElementBytes(
              policy.recurrent_state_storage);
      const auto cfg = *config;
      state_bytes = [cfg, kv_bytes, recurrent_bytes](std::uint64_t context) {
        return static_cast<std::uint64_t>(core::QwenDenseSessionStateBytes(
            cfg, context, kv_bytes, recurrent_bytes));
      };
      context_max = cfg.context_length;
      max_sessions = 8;
      base.notes.push_back(
          "Attention cache storage uses the shipped production policy "
          "(half precision). Not checked on this file.");
    } else if (route == "qwen36_a3b") {
      gufo::models::qwen36_a3b::Config config;
      std::string error;
      if (!gufo::models::qwen36_a3b::ReadConfig(reader, config, &error))
        return NoneYet("The qwen35moe header keys are not readable: " + error,
                       base);
      state_bytes = [config](std::uint64_t context) {
        return static_cast<std::uint64_t>(core::A3bSessionStateBytes(
            config.layers, config.full_attention_interval, config.KvDim(),
            config.ssm_conv_kernel, config.ConvChannels(), config.ssm_v_heads,
            config.ssm_head_dim, context));
      };
      max_sessions = 1;  // the server runs this model with one session
    } else {
      return NoneYet("Bubo has no memory estimate for this model yet.", base);
    }
  } catch (const std::exception& error) {
    return NoneYet(std::string("The state size overflowed: ") + error.what(),
                   base);
  }
  if (context_max < 2)
    return NoneYet("The header has no usable context length.", base);
  auto owned = MakeHeaderTable("header:" + facts.architecture, weights,
                               context_max, max_sessions, state_bytes);
  owned->table.cache_on_host = route == "qwen36_a3b";
  // A bound projector (an explicit --mmproj or the prepared projector of a
  // catalogue model) is a weights term, as in the Flash Next table.
  if (!resolved.artefacts.mmproj.empty() &&
      !resolved.artefacts.vision_disabled) {
    std::string error;
    const auto vision =
        QwenVisionUploadBytes(fs::u8path(resolved.artefacts.mmproj), &error);
    if (!vision)
      return NoneYet("The vision projector header cannot be read: " + error,
                     base);
    owned->table.vision_bytes = *vision;
    base.notes.push_back(
        "Vision projector " + resolved.artefacts.mmproj + ": " +
        std::to_string(*vision) +
        " bytes are in the weights term. The engine uploads the tower on the "
        "first image and reserves nothing for it at startup.");
  }
  auto planning = BuildReleaseMemoryEstimate(
      resolved, owned->package_id, request.available_bytes,
      request.host_available_bytes, &owned->table, request.capacity_budgets);
  const auto& inputs = planning.inputs;
  if (inputs.sessions > max_sessions)
    return NoneYet("This model runs with at most " +
                       std::to_string(max_sessions) + " session(s).",
                   base);
  base.evidence_label = "estimated from the file header";
  base.notes.push_back(kRamCacheNote);
  FillFitInputs(base, weights, facts.native_context, inputs.context,
                inputs.sessions, inputs.available_bytes);
  auto result = RunTable(std::move(planning), std::move(base));
  if (result.state == "estimated")
    FinishHeaderEstimate(result.memory_estimate, *owned, state_bytes);
  return result;
}

fs::path CatalogueEntryFolder(const CatalogueEntry& entry,
                              const fs::path& models_root) {
  return models_root / fs::u8path(entry.id);
}

CatalogueReadiness CatalogueReadinessFor(const CatalogueEntry& entry,
                                         const fs::path& models_root) {
  return CatalogueReadinessAt(entry, CatalogueEntryFolder(entry, models_root));
}

std::string CataloguePreparedState(const CatalogueEntry& entry,
                                   const fs::path& models_root) {
  return CatalogueReadinessFor(entry, models_root).state;
}

// The estimate of a catalogue package whose files are in `folder`. The folder
// need not be <models root>/<id>: an older download or an imported copy has
// the same layout under any name.
EstimateResult EstimateForCatalogueFolder(const CatalogueEntry& entry,
                                          const fs::path& folder,
                                          const EstimateRequest& request) {
  EstimateResult base;
  if (entry.flash_next_set) {
    const auto* reference = MemoryTableFor(entry.id);
    if (!reference)
      return NoneYet("No coefficient table for Flash Next.", base);
    const bool lowmemory =
        *entry.flash_next_set == prepare::ModelSet::LowMemory;
    auto coefficients = *reference;
    if (lowmemory && entry.measured_main_resident_bytes) {
      coefficients.package_id = entry.id;
      coefficients.main_weights_bytes = *entry.measured_main_resident_bytes;
    }
    const auto* table = &coefficients;
    const auto manifest = folder / "bubo-package.json";
    gufo::server::ReleaseResolution resolved;
    if (fs::exists(manifest) &&
        CatalogueReadinessAt(entry, folder).state == "prepared") {
      resolved = EstimateReleaseSettings(manifest, request, true);
    } else {
      // Before preparation, resolve the same profile against the catalogue's
      // package artefacts. A present invalid manifest never takes this path.
      gufo::server::ReleaseInputs release;
      release.profile = request.profile;
      release.architecture = entry.architecture;
      release.model_family =
          std::string(prepare::ModelFamilyForArchitecture(entry.architecture));
      release.native_context = table->context_max;
      release.manifest_present = true;
      release.environment = gufo::server::SnapshotReleaseEnvironment();
      release.real_environment = [](const char* name) {
        return std::getenv(name);
      };
      release.explicit_flags["--sessions"] = std::to_string(request.sessions);
      if (request.context)
        release.explicit_flags["--context"] = std::to_string(request.context);
      for (const auto& file : prepare::ModelCatalogue(*entry.flash_next_set))
        if (file.kind != prepare::Kind::MainShard)
          release.manifest_defaults[std::string(
              gufo::server::ModelArtefactSetting(file.kind))] =
              Utf8(folder / fs::u8path(file.destination));
      resolved = gufo::server::ResolveReleaseSettings(release);
    }
    auto planning = BuildPlanningReleaseMemoryEstimate(
        resolved, entry.id, request.available_bytes,
        request.host_available_bytes, table, request.capacity_budgets,
        request.context != 0);
    const auto& inputs = planning.inputs;
    base.evidence_label =
        lowmemory ? "estimated from the calibrated lower-memory coefficient"
                  : "estimated from the measured Flash Next coefficient";
    base.notes.push_back(
        "Only artefacts selected by the resolved release profile are counted.");
    base.notes.push_back(
        "Cache mode and allowance follow serve's resolved release settings. "
        "Cache ceilings do not allocate storage at startup; on-demand snapshot "
        "growth is excluded from both startup estimates.");
    if (*entry.flash_next_set == prepare::ModelSet::Original)
      base.notes.push_back(
          "The original set has the same shapes as the Huihui set. It was not "
          "measured separately.");
    if (!planning.value.is_null())
      FillFitInputs(base, planning.value.find("weights")->member_size("bytes"),
                    table->context_max, inputs.context, inputs.sessions,
                    inputs.available_bytes);
    return RunTable(std::move(planning), std::move(base));
  }
  if (!entry.estimate_pointers.empty()) {
    EstimateResult none;
    for (const auto& pointer : entry.estimate_pointers)
      none.notes.push_back("Source for a later merge: " + pointer);
    return NoneYet(entry.estimate_note.empty()
                       ? "This build has no memory estimate for this "
                         "architecture yet."
                       : entry.estimate_note,
                   std::move(none));
  }
  if (entry.packaging == "safetensors_folder")
    return EstimateForFolder(folder, request);
  const CatalogueFile* main_file = nullptr;
  for (const auto& file : entry.files)
    if (file.role == "main" && file.downloadable) {
      main_file = &file;
      break;
    }
  if (!main_file)
    return NoneYet("The entry has no main file.", base);
  const auto path = folder / fs::u8path(main_file->name);
  std::error_code ec;
  if (!fs::is_regular_file(path, ec) ||
      fs::file_size(path, ec) != main_file->bytes)
    return NoneYet(
        "The model file is not on this computer, so its header cannot be "
        "read.",
        base);
  auto inspection = InspectGguf(path);
  auto adjusted = request;
  // Model files only: a bound projector is its own term (EstimateForGguf).
  std::uint64_t weights = 0;
  for (const auto& file : entry.files)
    if (file.downloadable && file.role == "main")
      weights += file.bytes;
  adjusted.weights_bytes = weights;
  auto result = EstimateForGguf(inspection, adjusted);
  if (!entry.estimate_note.empty())
    result.notes.push_back(entry.estimate_note);
  return result;
}

EstimateResult EstimateForCatalogueEntry(const CatalogueEntry& entry,
                                         const fs::path& models_root,
                                         const EstimateRequest& request) {
  return EstimateForCatalogueFolder(
      entry, CatalogueEntryFolder(entry, models_root), request);
}

namespace {
// The package_id in a folder's bubo-package.json. Empty when the marker
// cannot be read or names no package.
std::string MarkerPackageId(const fs::path& folder) {
  try {
    return prepare::metadata::Parse(
               prepare::metadata::Read(folder / "bubo-package.json"))
        .member_str("package_id");
  } catch (const std::exception&) {
    return {};
  }
}
}  // namespace

EstimateResult EstimateForFolder(const fs::path& folder,
                                 const EstimateRequest& request) {
  EstimateResult base;
  if (fs::exists(folder / "bubo-package.json")) {
    // A package of a shipped catalogue entry is estimated as that entry is:
    // Flash Next keeps its measured table. A safetensors entry has no GGUF
    // package and is read below.
    const auto& shipped = ShippedCatalogue();
    if (const auto* entry =
            FindCatalogueEntry(shipped.entries, MarkerPackageId(folder));
        entry && entry->packaging != "safetensors_folder")
      return EstimateForCatalogueFolder(*entry, folder, request);
    const auto package =
        gufo::server::LoadModelPackage(folder, request.profile);
    // A DiffusionGemma folder's marker describes no GGUF package.
    if (!package.files.empty())
      return EstimateForGguf(InspectGguf(package.files.front().path), request);
  }
  std::string model_type;
  try {
    const auto config = prepare::metadata::Parse(
        prepare::metadata::Read(folder / "config.json"));
    model_type = config.member_str("model_type");
  } catch (const std::exception&) {
    return NoneYet("The folder has no readable config.json and no GGUF header.",
                   base);
  }
  if (model_type == "diffusion_gemma")
    return NoneYet(
        "DiffusionGemma keeps a canvas and a probability buffer per session. "
        "The loader has no validated memory estimate yet. A bound BF16 "
        "vision projector reserves 2 GiB for its tower and image-sized "
        "scratch at serve startup.",
        base);
  return NoneYet(
      "This build has no estimate for model_type '" + model_type + "'.", base);
}

Value EstimateJson(const EstimateResult& result) {
  auto value = Value::object();
  value["estimate_state"] = result.state;
  value["reason"] = OptionalString(result.reason);
  value["evidence_label"] = OptionalString(result.evidence_label);
  value["fit_verified"] = false;
  value["fit_inputs"] = result.fit_inputs;
  auto notes = Value::array();
  for (const auto& note : result.notes)
    notes.push_back(note);
  value["notes"] = std::move(notes);
  value["memory_estimate"] = result.memory_estimate;
  return value;
}

// ---- list ------------------------------------------------------------------

namespace {
// One unreadable package must not fail the whole listing: its row has no
// estimate, and the reason says why.
EstimateResult GuardedEstimate(const std::function<EstimateResult()>& run) {
  try {
    return run();
  } catch (const std::exception& error) {
    return NoneYet(std::string("The estimate failed: ") + error.what());
  }
}
bool Inside(const fs::path& file, const fs::path& folder) {
  std::error_code ec;
  const auto a = Lower(Utf8(fs::weakly_canonical(file, ec)));
  const auto b = Lower(Utf8(fs::weakly_canonical(folder, ec)));
  if (a.size() <= b.size() || a.compare(0, b.size(), b) != 0)
    return false;
  return a[b.size()] == '\\' || a[b.size()] == '/';
}
bool SameFolder(const fs::path& a, const fs::path& b) {
  std::error_code ec;
  return Lower(Utf8(fs::weakly_canonical(a, ec))) ==
         Lower(Utf8(fs::weakly_canonical(b, ec)));
}

// The --output folder that `bubo prepare` accepts for this package: the root
// above <root>/models/<id>, by prepare's own layout rule. Null when the folder
// is not in that layout (an older download under another name, an imported
// folder), so the GUI offers Repair only where prepare can act.
Value RepairOutput(const CatalogueEntry& entry, const fs::path& folder) {
  const auto root = folder.parent_path().parent_path();
  prepare::Options options;
  options.root = Utf8(root);
  options.catalogue_id = entry.id;
  if (root.empty() ||
      !SameFolder(fs::u8path(prepare::ModelRoot(options)), folder))
    return Value();
  return Value(Utf8(root));
}

// The `prepared_models` row of a catalogue package whose files are in
// `folder`. manifest_path: the package marker when it exists (Flash Next and
// plain catalogue models); model_path: the main model file. Either one starts
// the model (serve --manifest / --model). `state` is the size check:
// prepared, partial or not_prepared. `catalogue` is true: the entry gave the
// identity, the state and the capabilities.
Value CataloguePackageRow(const CatalogueEntry& entry, const fs::path& folder,
                          const std::string& state) {
  const auto manifest = folder / "bubo-package.json";
  std::error_code ec;
  std::string model_path;
  for (const auto& file : entry.files)
    if (file.downloadable && file.role == "main") {
      model_path = Utf8(fs::absolute(CatalogueFilePath(entry, file, folder), ec)
                            .lexically_normal());
      break;
    }
  auto row = PreparedModelRow(
      entry, fs::exists(manifest, ec) ? Utf8(manifest) : std::string(),
      model_path, state == "prepared");
  row["prepared_state"] = state;
  row["catalogue"] = true;
  row["repair_output"] = RepairOutput(entry, folder);
  return row;
}

// The row of a package whose id the shipped catalogue does not know, from its
// marker alone. Every file the marker lists must be present at its recorded
// size for the package to be prepared. `catalogue` is false.
Value MarkerPackageRow(const fs::path& folder, const Value& marker,
                       const std::string& id) {
  std::uint64_t size = 0;
  std::size_t listed = 0;
  bool complete = true, mtp = false, twin = false, mmproj = false;
  std::string model_path;
  const auto count = [&](const Value& item) {
    const auto name = item.member_str("file");
    if (name.empty())
      return fs::path();
    const auto path = folder / fs::u8path(name);
    const std::uint64_t bytes = item.member_size("size");
    std::error_code ec;
    ++listed;
    size += bytes;
    complete &=
        fs::is_regular_file(path, ec) && fs::file_size(path, ec) == bytes;
    return path;
  };
  if (const auto* model = marker.find("model"))
    if (const auto* files = model->find("files"); files && files->is_array())
      for (const auto& item : files->items()) {
        const auto path = count(item);
        if (model_path.empty() && !path.empty()) {
          std::error_code ec;
          model_path = Utf8(fs::absolute(path, ec).lexically_normal());
        }
      }
  if (const auto* artefacts = marker.find("artefacts");
      artefacts && artefacts->is_object())
    for (const auto& [key, item] : artefacts->members()) {
      count(item);
      mtp |= key == "mtp";
      twin |= key == "twin";
      mmproj |= Lower(key + item.member_str("file")).find("mmproj") !=
                std::string::npos;
    }
  auto row = Value::object();
  row["id"] = id;
  row["display_name"] =
      model_path.empty() ? id : Utf8(fs::u8path(model_path).filename());
  row["manifest_path"] = Utf8(folder / "bubo-package.json");
  row["model_path"] = OptionalString(model_path);
  row["size_bytes"] = size;
  row["prepared_state"] =
      listed == 0 ? "not_prepared" : (complete ? "prepared" : "partial");
  auto capabilities = Value::object();
  capabilities["context_max"] = Value();
  capabilities["mtp"] = mtp;
  capabilities["twin"] = twin;
  capabilities["mmproj"] = mmproj;
  row["capabilities"] = std::move(capabilities);
  row["catalogue"] = false;
  row["repair_output"] = Value();  // prepare knows catalogue packages only
  return row;
}
}  // namespace

ServeMemoryPolicy ResolveServeMemoryPolicy(
    gufo::server::ReleaseResolution& resolved, const std::string& package_id,
    const fs::path& model, std::uint64_t sessions,
    const PlanningMemoryBudgets& budgets, bool explicit_context,
    bool inspect_only, bool release_active) {
  const bool memory96 = Is96GiBStrixHalo(budgets) &&
                        resolved.model_family == "qwen3.8-flash-next";
  const auto estimate = [&] {
    auto result =
        BuildReleaseMemoryEstimate(resolved, package_id, budgets.gpu_available,
                                   budgets.host_available, nullptr, budgets);
    if (result.value.is_null() &&
        (memory96 || (release_active && sessions > 1))) {
      const auto fallback = EstimateForGguf(
          InspectGguf(model), {.profile = resolved.profile,
                               .context = result.inputs.context,
                               .sessions = sessions,
                               .available_bytes = budgets.gpu_available,
                               .host_available_bytes = budgets.host_available,
                               .capacity_budgets = budgets});
      result.value = fallback.memory_estimate;
      result.reason = fallback.reason;
    }
    return result;
  };
  auto planning = estimate();
  const auto refuse = [&] {
    const auto text = memory96 && !inspect_only
                          ? StartupMemoryRefusal(
                                budgets, planning.value,
                                package_id == prepare::kPackageId ||
                                    package_id == prepare::kOriginalPackageId)
                          : std::string{};
    if (text.empty())
      return;
    auto details = Value::object();
    auto& gpu = planning.value["gpu"];
    auto& host = planning.value["host"];
    auto needed = gpu["need"].member_size("bytes");
    auto available = gpu["available"].member_size("bytes");
    if (needed <= available && budgets.host_available) {
      const auto dedicated = budgets.gpu_dedicated.value_or(needed);
      needed = host["need"].member_size("bytes") +
               host["floor"].member_size("bytes") +
               (needed > dedicated ? needed - dedicated : 0);
      available = *budgets.host_available;
    }
    // The Go shortfall fields are uint64: exponent notation such as 2e+05
    // is valid JSON but cannot be decoded into those typed integer fields.
    const auto integer = [](std::uint64_t value) {
      return Value::parsed_number(static_cast<double>(value),
                                  std::to_string(value));
    };
    details["needed_bytes"] = integer(needed);
    details["available_bytes"] = integer(available);
    details["sessions"] = integer(sessions);
    details["context"] = integer(planning.inputs.context);
    throw ControlError(19, text, std::move(details));
  };
  ServeMemoryPolicy policy;
  if (release_active && (sessions > 1 || memory96) && !explicit_context) {
    bool viable = false;
    if (const auto* table = planning.value.find("recommended_context"))
      if (const auto* rows = table->find("by_sessions");
          rows && rows->is_array())
        for (const auto& row : rows->items())
          if (row.member_size("sessions") == sessions &&
              row.find("recommended") && row.find("recommended")->is_bool() &&
              row.find("recommended")->as_bool())
            viable = true;
    if (!viable)
      refuse();
    auto context_estimate = planning.value;
    if (context_estimate.is_null()) {
      context_estimate = Value::object();
      context_estimate["recommended_context_reason"] = planning.reason;
    }
    policy.context = gufo::server::ResolveDefaultSessionContext(
        resolved, sessions, context_estimate, memory96);
    if (policy.context)
      planning = estimate();
  }
  refuse();
  policy.estimate = std::move(planning.value);
  policy.estimate_reason = std::move(planning.reason);
  return policy;
}

std::string StartupMemoryRefusal(const PlanningMemoryBudgets& budgets,
                                 const Value& estimate, bool full_size) {
  const auto* fits = estimate.find("fits");
  if (!Is96GiBStrixHalo(budgets) || !fits || !fits->is_bool() ||
      fits->as_bool())
    return {};
  return full_size
             ? "The standard model does not fit the available memory. Choose "
               "the lower-memory model."
             : k96GbSetupMemoryRefusal;
}

Value BuildModelsList(const ModelsListOptions& options,
                      const std::vector<CatalogueEntry>& catalogue) {
  auto payload = Value::object();
  const auto budgets = ReadPlanningMemoryBudgets();
  const auto available =
      options.available_bytes ? options.available_bytes : budgets.gpu_available;
  EstimateRequest request;
  request.sessions = 1;
  // Explicit test budgets never mix with a live machine's capacity readings.
  request.capacity_budgets =
      options.available_bytes || options.host_available_bytes
          ? options.capacity_budgets
          : budgets;
  request.available_bytes = available;
  request.host_available_bytes = options.host_available_bytes
                                     ? options.host_available_bytes
                                     : budgets.host_available;

  auto listing = Value::object();
  listing["models_root"] = Utf8(options.models_root);
  listing["available_bytes"] = available ? Value(*available) : Value();
  listing["scan_depth_limit"] = kScanMaxDepth;
  listing["estimate_defaults"] = Value::object();
  listing["estimate_defaults"]["sessions"] = request.sessions;
  listing["estimate_defaults"]["context_tokens"] =
      "serve release-profile context (native when unset)";
  listing["catalogue_version"] = "2026-10-14";

  auto prepared = Value::array();
  auto catalogue_rows = Value::array();
  std::vector<fs::path> entry_folders;
  for (const auto& entry : catalogue) {
    auto row = CatalogueEntryJson(entry);
    const auto readiness = CatalogueReadinessFor(entry, options.models_root);
    const auto& state = readiness.state;
    row["prepared_state"] = state;
    const auto folder = CatalogueEntryFolder(entry, options.models_root);
    entry_folders.push_back(folder);
    row["local_folder"] = Utf8(folder);
    auto estimate = EstimateJson(GuardedEstimate([&] {
      return EstimateForCatalogueEntry(entry, options.models_root, request);
    }));
    // An incomplete model is a fact to report, not an error: name what is
    // missing so the caller can offer to resume.
    if (state == "partial")
      for (const auto& name : readiness.missing_files)
        estimate["notes"].push_back("Missing or wrong-sized file: " + name);
    if (Is96GiBStrixHalo(request.capacity_budgets) && entry.flash_next_set) {
      if (*entry.flash_next_set == prepare::ModelSet::LowMemory) {
        if (entry.download_enabled) {
          row["role"] = "recommended";
          row["default_status"] = "default";
        }
      } else {
        row["default_status"] = "none";
        row["download_enabled"] = false;
        const auto* memory = estimate.find("memory_estimate");
        const auto* gpu = memory ? memory->find("gpu") : nullptr;
        const auto* need = gpu ? gpu->find("need") : nullptr;
        const auto bytes = need ? need->member_size("bytes") : 0;
        row["disabled_reason"] = FullSize96GbMemoryRefusal(
            bytes ? std::optional<std::uint64_t>(bytes) : std::nullopt);
      }
    }
    row["estimate"] = std::move(estimate);
    catalogue_rows.push_back(row);
    if (state == "prepared")
      prepared.push_back(CataloguePackageRow(entry, folder, state));
  }

  ScanOptions scan;
  {
    // The default folder is scanned when it exists. A fresh install has none,
    // and that is not worth a warning.
    std::error_code ec;
    if (fs::is_directory(options.models_root, ec))
      scan.folders.push_back(options.models_root);
  }
  for (const auto& folder : options.scan_folders)
    scan.folders.push_back(folder);
  auto scanned = ScanModelFolders(scan);
  // A package folder found by the scan is one prepared_models row, wherever
  // it is. A catalogue folder under --models-root is already listed above and
  // is not listed again; the same id in two other folders gives two rows.
  for (const auto& folder : scanned.packages) {
    if (std::any_of(
            entry_folders.begin(), entry_folders.end(),
            [&](const fs::path& own) { return SameFolder(folder, own); }))
      continue;
    Value marker;
    try {
      marker = prepare::metadata::Parse(
          prepare::metadata::Read(folder / "bubo-package.json"));
    } catch (const std::exception&) {
    }
    const auto id = marker.member_str("package_id");
    if (id.empty()) {
      scanned.warnings.push_back(
          {Utf8(folder),
           "The package marker bubo-package.json cannot be read or has no "
           "package_id."});
      continue;
    }
    if (const auto* entry = FindCatalogueEntry(catalogue, id))
      prepared.push_back(CataloguePackageRow(
          *entry, folder, CatalogueReadinessAt(*entry, folder).state));
    else
      prepared.push_back(MarkerPackageRow(folder, marker, id));
  }
  auto scanned_rows = Value::array();
  std::size_t in_catalogue = 0;
  for (const auto& file : scanned.files) {
    bool packaged = false;
    for (const auto& folder : entry_folders)
      packaged |= Inside(file, folder);
    if (packaged) {
      ++in_catalogue;
      continue;
    }
    auto inspection = InspectGguf(file);
    if (inspection.facts.split_follower)
      continue;
    auto row = Value::object();
    const auto& facts = inspection.facts;
    row["path"] = facts.path;
    row["file_name"] = facts.file_name;
    row["folder"] = Utf8(file.parent_path());
    row["size_bytes"] = facts.total_bytes;
    row["shards_found"] = static_cast<std::uint64_t>(facts.shards_found);
    row["shards_expected"] = static_cast<std::uint64_t>(facts.shards_expected);
    row["architecture"] = OptionalString(facts.architecture);
    row["model_name"] = OptionalString(facts.model_name);
    row["native_context_tokens"] =
        facts.native_context ? Value(*facts.native_context) : Value();
    row["kind"] = !facts.header_ok             ? "unreadable"
                  : inspection.support.sidecar ? "sidecar"
                                               : "model";
    row["runnable"] = inspection.support.runnable;
    row["reason"] = OptionalString(inspection.support.reason);
    row["route"] = OptionalString(inspection.support.route);
    row["estimate"] = EstimateJson(
        GuardedEstimate([&] { return EstimateForGguf(inspection, request); }));
    scanned_rows.push_back(std::move(row));
  }
  listing["package_files_skipped"] = in_catalogue;
  auto scanned_folders = Value::array();
  for (const auto& folder : scanned.scanned_folders)
    scanned_folders.push_back(folder);
  listing["scanned_folders"] = std::move(scanned_folders);
  auto warnings = Value::array();
  for (const auto& warning : scanned.warnings) {
    auto row = Value::object();
    row["folder"] = warning.folder;
    row["reason"] = warning.reason;
    warnings.push_back(std::move(row));
  }
  payload["listing"] = std::move(listing);
  payload["prepared_models"] = std::move(prepared);
  payload["catalogue"] = std::move(catalogue_rows);
  payload["scanned"] = std::move(scanned_rows);
  payload["scan_warnings"] = std::move(warnings);
  return payload;
}

// ---- command ---------------------------------------------------------------

namespace {
bool ParseUnsigned(std::string_view text, std::uint64_t& out) {
  if (text.empty() || text.size() > 12 ||
      !std::all_of(text.begin(), text.end(),
                   [](unsigned char c) { return std::isdigit(c) != 0; }))
    return false;
  out = std::stoull(std::string(text));
  return true;
}
}  // namespace

bool ParseModelsCommand(std::span<const char* const> args,
                        ModelsCommandOptions& options, std::string& error) {
  if (args.empty()) {
    error = "missing models verb (list or estimate)";
    return false;
  }
  options.verb = args[0];
  if (options.verb == "--help" || options.verb == "help") {
    options.help = true;
    return true;
  }
  if (options.verb != "list" && options.verb != "estimate") {
    error = "unknown models verb '" + options.verb + "'";
    return false;
  }
  gufo::cli::ArgParser parser("bubo models");
  parser.SetImplicitHelp(false).SetInlineValues(false);
  const auto missing = [](std::string_view flag) {
    if (flag == "--model")
      return std::string("--model requires a path or a catalogue id");
    if (flag == "--context" || flag == "--sessions")
      return std::string(flag) + " requires a positive number";
    return std::string(flag) + " requires a folder";
  };
  parser.SetErrorHandler([&](auto kind, std::string_view arg) {
    return kind == gufo::cli::ArgParser::ParseError::MissingValue
               ? missing(arg)
               : "unknown argument '" + std::string(arg) + "'";
  });
  parser.AddFlag("", "--json", "Machine output", "Output", &options.machine);
  parser.AddFlag("", "--jsonl", "Machine stream", "Output", &options.machine);
  parser.AddFlag("", "--help", "Print help", "General", &options.help);
  const auto path = [&](std::string_view flag, auto store) {
    parser.AddCustomOption("", flag, "PATH", "Path", "Model",
                           [&, store](auto name, auto value, std::string* err) {
                             if (value.empty()) {
                               *err = missing(name);
                               return false;
                             }
                             store(value);
                             return true;
                           });
  };
  path("--models-root",
       [&](auto value) { options.models_root = fs::u8path(value); });
  if (options.verb == "list")
    path("--scan-folder", [&](auto value) {
      options.scan_folders.push_back(fs::u8path(value));
    });
  else {
    path("--model", [&](auto value) { options.model = value; });
    parser.AddOption("", "--profile", "PROFILE", "Serve release profile",
                     "Estimate", &options.profile);
    for (const auto* flag : {"--context", "--sessions"})
      parser.AddCustomOption(
          "", flag, "N", "Positive number", "Estimate",
          [&](auto name, auto value, std::string* err) {
            std::uint64_t number = 0;
            if (!ParseUnsigned(value, number) || number == 0) {
              *err = missing(name);
              return false;
            }
            (name == "--context" ? options.context : options.sessions) = number;
            return true;
          });
  }
  if (!parser.Parse(args.subspan(1), &error))
    return false;
  if (options.verb == "estimate" && options.model.empty() && !options.help) {
    error = "estimate requires --model PATH or --model CATALOGUE_ID";
    return false;
  }
  return true;
}

namespace {
void PrintHelp(std::ostream& out) {
  out << "Usage: bubo models list [--json|--jsonl] [--scan-folder PATH]... "
         "[--models-root PATH]\n"
         "       bubo models estimate --model PATH|CATALOGUE_ID "
         "[--json|--jsonl] "
         "[--profile PROFILE] [--context N] [--sessions N]\n\n"
         "list: prepared packages, catalogue entries and GGUF files in the\n"
         "model folder and in each --scan-folder (depth limit "
      << kScanMaxDepth
      << ", never a drive root,\n"
         "links not followed). A folder with a bubo-package.json is one "
         "prepared\n"
         "package; its files are not listed one by one. Only file headers are\n"
         "read.\n"
         "estimate: memory estimate and recommended context. No server "
         "starts.\n";
}
std::string Gib(std::uint64_t bytes, int digits = 1) {
  char text[32];
  std::snprintf(text, sizeof(text), "%.*f GiB", digits, BytesToGiB(bytes));
  return text;
}
}  // namespace

int RunModels(std::span<const char* const> args,
              std::string_view local_app_data, std::ostream& out,
              std::ostream& err) {
  ModelsCommandOptions options;
  std::string parse_error;
  const bool machine = MachineRequested(args);
  const bool parsed = ParseModelsCommand(args, options, parse_error);
  if (parsed && options.help && !machine) {
    PrintHelp(out);
    return 0;
  }
  std::unique_ptr<MachineWriter> writer;
  if (machine)
    writer = std::make_unique<MachineWriter>(out);
  if (!parsed || options.help) {
    if (writer) {
      if (options.help && parsed)
        writer->Write("capabilities", Capabilities());
      else
        writer->Write("error", Error("invalid_arguments", "models", "arguments",
                                     false, "Run bubo models --help."));
      return parsed ? 0 : 2;
    }
    err << "models: " << parse_error << "; run bubo models --help\n";
    return 2;
  }
  if (options.models_root.empty()) {
    if (local_app_data.empty()) {
      if (writer)
        writer->Write("error", Error("invalid_arguments", "models", "arguments",
                                     false, "Use --models-root PATH."));
      else
        err << "models: LOCALAPPDATA is unavailable; use --models-root\n";
      return 2;
    }
    options.models_root =
        fs::u8path(std::string(local_app_data)) / "Bubo" / "models";
  }
  const auto& shipped = ShippedCatalogue();
  if (!shipped.error.empty()) {
    if (writer)
      writer->Write("error", Error("unexpected_failure", "models", "catalogue",
                                   false, "Reinstall Bubo."));
    else
      err << "models: the shipped catalogue is invalid: " << shipped.error
          << '\n';
    return 1;
  }
  try {
    if (options.verb == "list") {
      ModelsListOptions list;
      list.models_root = options.models_root;
      list.scan_folders = options.scan_folders;
      auto payload = BuildModelsList(list, shipped.entries);
      if (writer) {
        if (MachineMode::JsonLines == ParseMachineMode(args))
          writer->Write("capabilities", Capabilities());
        return writer->Write("models", std::move(payload)) ? 0 : 1;
      }
      out << "Prepared models:\n";
      for (const auto& row : payload["prepared_models"].items()) {
        // The same id can come from two folders: name the folder and say so
        // when the files are not all there.
        out << "  " << row.member_str("id") << "  "
            << Gib(row.member_size("size_bytes"));
        if (row.member_str("prepared_state") != "prepared")
          out << "  (" << row.member_str("prepared_state") << ')';
        if (const auto path = row.member_str("manifest_path"); !path.empty())
          out << "  " << path;
        out << '\n';
      }
      out << "Catalogue:\n";
      for (const auto& row : payload["catalogue"].items())
        out << "  " << row.member_str("id") << "  "
            << row.member_str("prepared_state") << "  "
            << Gib(row.member_size("download_bytes")) << "  "
            << row.find("estimate")->member_str("estimate_state") << '\n';
      out << "On this computer:\n";
      for (const auto& row : payload["scanned"].items())
        out << "  " << row.member_str("path") << "  "
            << (row.find("runnable")->as_bool() ? "runnable" : "not runnable")
            << '\n';
      for (const auto& row : payload["scan_warnings"].items())
        out << "  note: " << row.member_str("folder") << ": "
            << row.member_str("reason") << '\n';
      return 0;
    }
    // estimate
    EstimateRequest request;
    request.profile = options.profile;
    request.context = options.context;
    request.sessions = options.sessions;
    const auto budgets = ReadPlanningMemoryBudgets();
    request.capacity_budgets = budgets;
    request.available_bytes = budgets.gpu_available;
    request.host_available_bytes = budgets.host_available;
    EstimateResult result;
    std::string subject = options.model;
    std::string kind = "path";
    if (const auto* entry =
            FindCatalogueEntry(shipped.entries, options.model)) {
      kind = "catalogue";
      result = EstimateForCatalogueEntry(*entry, options.models_root, request);
    } else {
      const auto path = fs::u8path(options.model);
      std::error_code ec;
      if (fs::is_directory(path, ec))
        result = EstimateForFolder(path, request);
      else if (fs::is_regular_file(path, ec))
        result = EstimateForGguf(InspectGguf(path), request);
      else {
        if (writer)
          writer->Write("error",
                        Error("model_missing", "models", "model", false,
                              "Use a file, a folder or a catalogue "
                              "id from bubo models list."));
        else
          err << "models: '" << options.model
              << "' is not a file, a folder or a catalogue id\n";
        return 12;
      }
    }
    auto payload = EstimateJson(result);
    payload["subject"] = subject;
    payload["subject_kind"] = kind;
    if (writer) {
      if (MachineMode::JsonLines == ParseMachineMode(args))
        writer->Write("capabilities", Capabilities());
      return writer->Write("model_estimate", std::move(payload)) ? 0 : 1;
    }
    out << "Model: " << subject << '\n';
    if (result.state != "estimated") {
      out << "No estimate yet: " << result.reason << '\n';
      return 0;
    }
    const auto& estimate = result.memory_estimate;
    out << "Evidence: " << result.evidence_label << '\n';
    out << "At least: "
        << Gib(estimate.find("total_range")->find("min")->member_size("bytes"),
               0)
        << " at " << estimate.member_size("context_tokens") << " tokens, "
        << estimate.member_size("sessions") << " session(s)\n";
    if (const auto* rec = estimate.find("recommended_context");
        rec && !rec->is_null())
      for (const auto& row : rec->find("by_sessions")->items())
        out << "  " << row.member_size("sessions")
            << " session(s): " << row.member_size("context_per_session")
            << " tokens"
            << (row.find("fits")->as_bool() ? "" : " (does not fit)") << '\n';
    out << "Fit is not verified.\n";
    for (const auto& note : result.notes)
      out << "Note: " << note << '\n';
    return 0;
  } catch (const std::exception& error) {
    if (writer)
      writer->Write("error", Error("unexpected_failure", "models", "scan",
                                   false, "Run the command again."));
    else
      err << "models: " << error.what() << '\n';
    return 1;
  }
}

}  // namespace bubo::cli
