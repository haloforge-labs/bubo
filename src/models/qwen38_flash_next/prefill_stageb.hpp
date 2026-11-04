#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_PREFILL_STAGEB_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_PREFILL_STAGEB_HPP_

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>

#include "src/core/json.hpp"
#include "src/models/qwen38_flash_next/prefill_experts.hpp"
#include "src/models/qwen38_flash_next/prefill_stageb_types.hpp"

namespace gufo::models::qwen38_flash_next {
struct PrefillStageBSettings {
  bool enabled{false};
  std::filesystem::path directory;
  std::array<bool, 48> layers{};
  std::uint32_t stride{1}, sample_every{64};
};

inline std::uint32_t StageBPositive(const char* text, std::uint32_t fallback) {
  if (!text)
    return fallback;
  std::string_view value(text);
  std::uint32_t result = 0;
  const auto parsed =
      std::from_chars(value.data(), value.data() + value.size(), result);
  if (!result || parsed.ec != std::errc{} ||
      parsed.ptr != value.data() + value.size())
    throw std::invalid_argument(
        "invalid GUFO_PREFILL_STAGEB sampling interval");
  return result;
}

inline PrefillStageBSettings ParsePrefillStageB(const char* directory,
                                                const char* layers,
                                                const char* stride,
                                                const char* sample_every) {
  PrefillStageBSettings settings;
  if (!directory)
    return settings;  // No allocation or auxiliary parsing off.
  const std::string_view path(directory);
  if (path.size() < 3 || (path[0] != 'S' && path[0] != 's') || path[1] != ':' ||
      (path[2] != '\\' && path[2] != '/'))
    throw std::invalid_argument(
        "GUFO_PREFILL_STAGEB requires an absolute S: directory");
  settings.directory = std::filesystem::path(directory).lexically_normal();
  settings.enabled = true;
  settings.stride = StageBPositive(stride, 1);
  settings.sample_every = StageBPositive(sample_every, 64);
  if (!layers)
    settings.layers.fill(true);
  else {
    std::string_view remaining(layers);
    for (;;) {
      const auto comma = remaining.find(',');
      const auto part = remaining.substr(0, comma);
      unsigned layer = 48;
      const auto parsed =
          std::from_chars(part.data(), part.data() + part.size(), layer);
      if (part.empty() || parsed.ec != std::errc{} ||
          parsed.ptr != part.data() + part.size() || layer >= 48 ||
          settings.layers[layer])
        throw std::invalid_argument("invalid GUFO_PREFILL_STAGEB_LAYERS");
      settings.layers[layer] = true;
      if (comma == std::string_view::npos)
        break;
      remaining.remove_prefix(comma + 1);
    }
  }
  return settings;
}

inline const PrefillStageBSettings& GetPrefillStageB() {
  static const auto settings = [] {
    const char* dir = std::getenv("GUFO_PREFILL_STAGEB");
    return ParsePrefillStageB(
        dir, dir ? std::getenv("GUFO_PREFILL_STAGEB_LAYERS") : nullptr,
        dir ? std::getenv("GUFO_PREFILL_STAGEB_STRIDE") : nullptr,
        dir ? std::getenv("GUFO_PREFILL_STAGEB_ROWS") : nullptr);
  }();
  return settings;
}

inline void RequireStageBReference(const PrefillExpertsSettings& experts) {
  const bool unpruned =
      experts.layer_keep.empty()
          ? experts.keep == 10
          : std::all_of(experts.layer_keep.begin(), experts.layer_keep.end(),
                        [](auto keep) { return keep == 10; });
  if (experts.active &&
      (!unpruned || (experts.mode != PrefillExpertsMode::kRenorm &&
                     experts.mode != PrefillExpertsMode::kRef10)))
    throw std::invalid_argument(
        "stage B requires an unpruned top-10 reference (knob unset, renorm:10 "
        "or ref10:10)");
}

inline std::uint32_t StageBCount(std::uint32_t start, std::uint32_t rows,
                                 std::uint32_t every) {
  const auto first = (std::uint64_t{start} + every - 1) / every;
  const auto end = (std::uint64_t{start} + rows + every - 1) / every;
  return static_cast<std::uint32_t>(end - first);
}
inline std::uint64_t NextStageBPass() {
  static std::atomic<std::uint64_t> sequence{0};
  return ++sequence;
}

// Sample-only storage precision. Dot products and fold derivation remain FP64.
// Positions/layers are losslessly implied by each layer's offset and interval.
// 16 uint16 fields = 32 bytes per sampled token/layer (<8 MiB for the
// manifest). Gates, L2, max and masses use UNORM16 (ranges 2,4,2,1
// respectively). Five nonnegative norms use round-to-nearest-even BF16
// (relative error <=1/256 for normal values). top4 is reconstructed as
// 1-omitted_mass[k4].
inline std::array<std::uint16_t, 16> PackStageBRow(const StageBRow& row) {
  std::array<std::uint16_t, 16> packed{};
  const auto unorm = [](float value, float maximum) {
    if (!std::isfinite(value) || value < -1e-5f || value > maximum + 1e-5f)
      throw std::runtime_error("stage B sample outside UNORM range");
    return static_cast<std::uint16_t>(
        std::lround(std::clamp(value, 0.f, maximum) / maximum * 65535.));
  };
  for (unsigned s = 0; s < 4; ++s)
    packed[s] = unorm(row.gates[s], 2);
  packed[4] = unorm(row.gate_l2, 4);
  packed[5] = unorm(row.gate_max, 2);
  packed[6] = unorm(row.top1_mass, 1);
  for (unsigned n = 0; n < 5; ++n) {
    const float value = n == 0 ? row.f_norm : row.error_norm[n - 1];
    if (!std::isfinite(value) || value < 0)
      throw std::runtime_error("invalid stage B sample norm");
    const auto bits = std::bit_cast<std::uint32_t>(value);
    const auto rounded = (bits + 0x7fffU + ((bits >> 16) & 1U)) >> 16;
    if (rounded >= 0x7f80U)
      throw std::runtime_error("stage B sample BF16 overflow");
    packed[7 + n] = static_cast<std::uint16_t>(rounded);
  }
  for (unsigned k = 0; k < 4; ++k)
    packed[12 + k] = unorm(row.omitted[k], 1);
  return packed;
}

class PrefillStageBPass {
public:
  PrefillStageBPass(const PrefillStageBSettings& settings, std::uint64_t pass,
                    std::uint32_t start, std::uint32_t tokens)
      : settings_(settings), pass_(pass), start_(start), tokens_(tokens) {
    if (!settings.enabled || !tokens ||
        std::uint64_t{start} + tokens > UINT32_MAX)
      throw std::invalid_argument("invalid stage B pass");
    // Same filesystem/checked binary-stream layer as prefill telemetry.
    std::filesystem::create_directories(settings.directory);
    static std::atomic<std::uint64_t> unique{0};
    do {
      relative_ = "stageb-pass-" + std::to_string(pass) + "-" +
                  std::to_string(++unique);
    } while (
        !std::filesystem::create_directory(settings.directory / relative_));
  }
  bool Selected(unsigned layer) const {
    if (layer >= 48)
      throw std::out_of_range("invalid stage B layer");
    return settings_.layers[layer];
  }
  std::uint32_t SampleCapacity() const {
    return std::max(1U, StageBCount(start_, tokens_, settings_.sample_every));
  }
  void ObserveLayer(unsigned layer, std::uint32_t start, std::uint32_t rows) {
    if (!Selected(layer) || observed_[layer] || start < start_ || !rows ||
        std::uint64_t{start} + rows != std::uint64_t{start_} + tokens_ ||
        (start != start_ && layer != 47))
      throw std::runtime_error("invalid stage B layer coverage");
    observed_[layer] = true;
    row_start_[layer] = start;
    rows_[layer] = rows;
  }
  void Finish(std::span<const StageBTotal> totals,
              std::span<const StageBRow> samples) {
    if (finished_ || totals.size() != 48 ||
        samples.size() != 48 * std::size_t{SampleCapacity()})
      throw std::runtime_error("incomplete stage B buffers/pass");
    auto selected = json::Value::array();
    for (unsigned l = 0; l < 48; ++l) {
      if (Selected(l)) {
        if (!observed_[l])
          throw std::runtime_error("missing stage B layer");
        selected.push_back(static_cast<std::size_t>(l));
        ValidateTotal(totals[l],
                      StageBCount(row_start_[l], rows_[l], settings_.stride));
      }
    }
    static_assert(std::endian::native == std::endian::little);
    std::ofstream binary(settings_.directory / relative_ / "rows.bin",
                         std::ios::binary);
    binary.write("GUFOPFB2", 8);
    std::uint64_t offset = 8;
    std::string lines;
    for (unsigned l = 0; l < 48; ++l)
      if (Selected(l)) {
        const auto count =
            StageBCount(row_start_[l], rows_[l], settings_.sample_every);
        const auto* data = samples.data() + std::size_t{l} * SampleCapacity();
        const auto first =
            (std::uint64_t{row_start_[l]} + settings_.sample_every - 1) /
            settings_.sample_every;
        for (unsigned i = 0; i < count; ++i) {
          if (data[i].layer != l ||
              data[i].token != (first + i) * settings_.sample_every)
            throw std::runtime_error("invalid stage B sample position");
          // Validate the complete GPU record before compacting sample fields.
          const auto fields = std::bit_cast<std::array<float, 19>>(data[i]);
          for (unsigned f = 2; f < fields.size(); ++f)
            if (!std::isfinite(fields[f]))
              throw std::runtime_error("nonfinite stage B sample");
        }
        for (unsigned i = 0; i < count; ++i) {
          const auto packed = PackStageBRow(data[i]);
          binary.write(reinterpret_cast<const char*>(packed.data()), 32);
        }
        auto record = json::Value::object();
        record["schema"] = "gufo-prefill-stageb-v1";
        record["pass_index"] = pass_;
        record["start"] = static_cast<std::size_t>(start_);
        record["tokens"] = static_cast<std::size_t>(tokens_);
        record["layer"] = static_cast<std::size_t>(l);
        record["row_start"] = static_cast<std::size_t>(row_start_[l]);
        record["rows_available"] = static_cast<std::size_t>(rows_[l]);
        record["rows"] = totals[l].rows;
        record["stride"] = static_cast<std::size_t>(settings_.stride);
        record["sample_every"] =
            static_cast<std::size_t>(settings_.sample_every);
        record["selected_layers"] = selected;
        record["router_weight_basis"] = "original-top10-normalized";
        record["top1_mass"] = totals[l].top1_mass;
        record["top4_mass"] = totals[l].top4_mass;
        record["gate_l2"] = totals[l].gate_l2;
        record["row_file"] = (relative_ / "rows.bin").generic_string();
        record["row_offset"] = offset;
        record["row_count"] = static_cast<std::size_t>(count);
        record["record_bytes"] = 32;
        record["row_layout"] =
            "little-endian <16H: UNORM16 gates[4]/2,gate_l2/4,gate_max/2,top1; "
            "BF16 f_norm,error_norm[k2,k3,k4,k6]; UNORM16 "
            "omitted_mass[k2,k3,k4,k6]; top4=1-omitted[k4]; token/layer from "
            "row_start, sample_every and layer metadata";
        record["k"] = json::Value::object();
        for (unsigned k = 0; k < 5; ++k) {
          auto value = json::Value::object();
          value["rf"] = totals[l].rf[k];
          value["rr"] = totals[l].rr[k];
          value["ff"] = totals[l].ff;
          value["error2"] = totals[l].error2[k];
          value["omitted_mass"] = totals[l].omitted[k];
          record["k"][std::to_string(kStageBKeep[k])] = std::move(value);
        }
        lines += record.dump() + '\n';
        offset += std::uint64_t{count} * 32;
      }
    binary.flush();
    binary.close();
    if (!binary)
      throw std::runtime_error("stage B sample write/close failed");
    std::ofstream out(settings_.directory / "stageb-totals.jsonl",
                      std::ios::binary | std::ios::app);
    out << lines;
    out.flush();
    out.close();
    if (!out)
      throw std::runtime_error("stage B totals write/close failed");
    finished_ = true;
  }

private:
  static void ValidateTotal(const StageBTotal& t, unsigned expected) {
    const auto finite = [](double v) {
      if (!std::isfinite(v))
        throw std::runtime_error("nonfinite stage B total");
    };
    if (t.rows != expected || t.invalid)
      throw std::runtime_error("invalid stage B row count/input");
    finite(t.ff);
    finite(t.top1_mass);
    finite(t.top4_mass);
    finite(t.gate_l2);
    for (unsigned k = 0; k < 5; ++k) {
      finite(t.rf[k]);
      finite(t.rr[k]);
      finite(t.error2[k]);
      finite(t.omitted[k]);
    }
  }
  const PrefillStageBSettings& settings_;
  std::uint64_t pass_;
  std::uint32_t start_, tokens_;
  std::array<bool, 48> observed_{};
  std::array<std::uint32_t, 48> row_start_{}, rows_{};
  std::filesystem::path relative_;
  bool finished_{false};
};

inline std::unique_ptr<PrefillStageBPass> MakePrefillStageBPass(
    const PrefillStageBSettings& settings, std::uint64_t pass,
    std::uint32_t start, std::uint32_t tokens) {
  if (!settings.enabled)
    return nullptr;
  return std::make_unique<PrefillStageBPass>(settings, pass, start, tokens);
}
}  // namespace gufo::models::qwen38_flash_next
#endif
