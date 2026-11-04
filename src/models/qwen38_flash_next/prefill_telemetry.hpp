#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_PREFILL_TELEMETRY_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_PREFILL_TELEMETRY_HPP_

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/json.hpp"

namespace gufo::models::qwen38_flash_next {

struct PrefillTelemetrySettings {
  bool enabled{false};
  std::filesystem::path directory;
  std::array<bool, 48> layers{};
};

inline PrefillTelemetrySettings ParsePrefillTelemetry(const char* directory,
                                                      const char* layers) {
  PrefillTelemetrySettings result;
  if (directory == nullptr)
    return result;  // No path/string allocation or auxiliary parsing while off.
  if (*directory == '\0')
    throw std::invalid_argument("GUFO_PREFILL_TELEMETRY needs a directory");
  result.enabled = true;
  result.directory = directory;
  std::string_view text(layers ? layers : "3,15,27,39,47");
  do {
    const auto end = text.find(',');
    const auto part = text.substr(0, end);
    unsigned layer = 48;
    const auto parsed =
        std::from_chars(part.data(), part.data() + part.size(), layer);
    if (part.empty() || parsed.ec != std::errc{} ||
        parsed.ptr != part.data() + part.size() || layer >= 48 ||
        result.layers[layer])
      throw std::invalid_argument("invalid GUFO_PREFILL_TELEMETRY_LAYERS");
    result.layers[layer] = true;
    if (end == std::string_view::npos)
      break;
    text.remove_prefix(end + 1);
  } while (true);
  return result;
}

inline const PrefillTelemetrySettings& GetPrefillTelemetry() {
  static const auto settings = [] {
    const char* directory = std::getenv("GUFO_PREFILL_TELEMETRY");
    return ParsePrefillTelemetry(
        directory,
        directory ? std::getenv("GUFO_PREFILL_TELEMETRY_LAYERS") : nullptr);
  }();
  return settings;
}

struct TelemetryExpertPopulation {
  std::uint32_t touched{0}, p50{0}, p90{0};
};

inline TelemetryExpertPopulation TelemetryPopulation(
    std::span<const std::uint32_t> counts) {
  std::vector<std::uint32_t> touched;
  for (auto count : counts)
    if (count)
      touched.push_back(count);
  if (touched.empty())
    return {};
  std::sort(touched.begin(), touched.end());
  // Nearest rank, over touched experts only, including a ragged bucket.
  return {static_cast<std::uint32_t>(touched.size()),
          touched[(touched.size() * 50 + 99) / 100 - 1],
          touched[(touched.size() * 90 + 99) / 100 - 1]};
}

inline std::uint64_t PackedRowTiles(std::span<const std::uint32_t> counts,
                                    std::uint32_t tile_rows) {
  if (!tile_rows)
    throw std::invalid_argument("telemetry tile rows must be positive");
  std::uint64_t tiles = 0;
  for (auto count : counts)
    tiles += (std::uint64_t{count} + tile_rows - 1) / tile_rows;
  return tiles;
}

struct TelemetryRecord {
  std::uint32_t token{0}, layer{0};
  // Original top-10 normalized masses, effective write gates (not partial
  // logits), and L2 over the four residual-stream gates.
  std::array<float, 7> values{};  // top1, top4, L2, gate0, gate1, gate2, gate3
};
static_assert(sizeof(TelemetryRecord) == 36);
static_assert(std::endian::native == std::endian::little);

inline TelemetryRecord MakeTelemetryRecord(std::uint32_t token,
                                           std::uint32_t layer,
                                           std::span<const float> weights,
                                           std::span<const float> inject,
                                           std::uint32_t parts) {
  if (weights.size() != 10 || parts == 0 || inject.size() != 4 * parts)
    throw std::invalid_argument("invalid telemetry router/inject shape");
  TelemetryRecord record{token, layer, {}};
  record.values[0] = weights[0];
  for (unsigned i = 0; i < 4; ++i)
    record.values[1] += weights[i];
  float norm = 0;
  for (unsigned stream = 0; stream < 4; ++stream) {
    float logit = 0;
    for (unsigned p = 0; p < parts; ++p)
      logit += inject[stream * parts + p];
    const float gate = 2.0f / (1.0f + std::exp(-logit / 4.0f));
    record.values[3 + stream] = gate;
    norm += gate * gate;
  }
  record.values[2] = std::sqrt(norm);
  for (float value : record.values)
    if (!std::isfinite(value))
      throw std::runtime_error("nonfinite prefill telemetry value");
  return record;
}

inline std::uint64_t NextTelemetryPass() {
  static std::atomic<std::uint64_t> sequence{0};
  return ++sequence;
}

class PrefillTelemetryPass {
public:
  PrefillTelemetryPass(const PrefillTelemetrySettings& settings,
                       std::uint64_t pass, std::uint32_t start,
                       std::uint32_t tokens, std::string policy)
      : settings_(settings),
        pass_(pass),
        start_(start),
        tokens_(tokens),
        policy_(std::move(policy)) {
    if (!settings.enabled || tokens == 0 ||
        std::uint64_t{start} + tokens > UINT32_MAX)
      throw std::invalid_argument("invalid telemetry pass");
    std::filesystem::create_directories(settings.directory);
    // Exclusive directory creation also prevents overwrites across processes,
    // restarts and separate executors sharing an output directory.
    static std::atomic<std::uint64_t> unique{0};
    do {
      directory_ = settings.directory / ("pass-" + std::to_string(pass) + "-" +
                                         std::to_string(++unique));
    } while (!std::filesystem::create_directory(directory_));
    binary_.open(directory_ / "tokens.bin", std::ios::binary);
    layers_.open(directory_ / "layers.jsonl", std::ios::binary);
    CheckFiles();
    binary_.write("GUFOPFT1", 8);
    const std::array<std::uint32_t, 4> format{1, 32, 36, 0};
    binary_.write(reinterpret_cast<const char*>(format.data()), 16);
    binary_.write(reinterpret_cast<const char*>(&pass_), 8);
    CheckFiles();
  }
  bool Selected(std::uint32_t layer) const {
    if (layer >= settings_.layers.size())
      throw std::out_of_range("telemetry layer outside supported trunk");
    return settings_.layers[layer];
  }
  void RecordLayer(std::uint32_t layer, std::uint32_t row_start,
                   std::uint32_t rows, std::span<const std::uint32_t> counts,
                   std::string_view route, std::uint32_t gate_tile_rows,
                   std::uint32_t down_tile_rows, unsigned gate_dispatches,
                   std::uint64_t gate_tiles, std::uint64_t down_tiles,
                   json::Value dispatches = json::Value::array()) {
    if (finished_ || layer >= population_layers_.size() ||
        population_layers_[layer] || rows == 0 || row_start < start_ ||
        std::uint64_t{row_start} + rows != std::uint64_t{start_} + tokens_ ||
        (row_start != start_ && layer != 47))
      throw std::runtime_error("invalid telemetry layer coverage");
    if (route != "mmq" && route != "lanerow" && route != "f16-wmma-iq-paired" &&
        route != "iq-pair2" && route != "f16-wmma-paired" &&
        route != "f16-wmma-separate")
      throw std::runtime_error("unknown telemetry route");
    const auto population = TelemetryPopulation(counts);
    const bool packed = route != "mmq";
    if (packed && (PackedRowTiles(counts, gate_tile_rows) != gate_tiles ||
                   PackedRowTiles(counts, down_tile_rows) != down_tiles))
      throw std::runtime_error("telemetry tile map/count mismatch");
    auto record = json::Value::object();
    record["pass"] = pass_;
    record["layer"] = static_cast<std::size_t>(layer);
    record["row_start"] = static_cast<std::size_t>(row_start);
    record["rows"] = static_cast<std::size_t>(rows);
    record["distinct_experts"] = static_cast<std::size_t>(population.touched);
    record["rows_p50"] = static_cast<std::size_t>(population.p50);
    record["rows_p90"] = static_cast<std::size_t>(population.p90);
    record["quantile"] = "nearest-rank-touched";
    record["route"] = std::string(route);
    record["gate_tile_rows"] = static_cast<std::size_t>(gate_tile_rows);
    record["down_tile_rows"] = static_cast<std::size_t>(down_tile_rows);
    record["gate_dispatches"] = static_cast<std::size_t>(gate_dispatches);
    record["gate_tiles"] = gate_tiles;
    record["down_tiles"] = down_tiles;
    record["packed_tiles_launched"] =
        packed ? gate_tiles * gate_dispatches + down_tiles : 0;
    std::uint64_t row_tiles =
        packed ? gate_tiles * gate_dispatches + down_tiles : 0;
    for (const auto& dispatch : dispatches.items()) {
      const auto* tiles = dispatch.find("row_tiles_launched");
      if (!tiles)
        throw std::runtime_error("MMQ dispatch has no tile observation");
      row_tiles += tiles->as_size();
    }
    record["row_tiles_launched"] = row_tiles;
    record["dispatches"] = std::move(dispatches);
    layers_ << record.dump() << '\n';
    CheckFiles();
    population_layers_[layer] = true;
    if (Selected(layer))
      expected_records_ += rows;
  }
  void Record(const TelemetryRecord& record) {
    if (finished_ || !Selected(record.layer) || record.token < start_ ||
        std::uint64_t{record.token} >= std::uint64_t{start_} + tokens_)
      throw std::runtime_error("telemetry record outside pass");
    for (float value : record.values)
      if (!std::isfinite(value))
        throw std::runtime_error("nonfinite telemetry record");
    binary_.write(reinterpret_cast<const char*>(&record), sizeof(record));
    ++records_;
    CheckFiles();
  }
  std::filesystem::path Finish() {
    if (finished_)
      throw std::logic_error("telemetry pass already finalized");
    if (!std::all_of(population_layers_.begin(), population_layers_.end(),
                     [](bool present) { return present; }) ||
        records_ != expected_records_)
      throw std::runtime_error("incomplete prefill telemetry pass");
    binary_.flush();
    layers_.flush();
    CheckFiles();
    binary_.close();
    layers_.close();
    CheckFiles();
    auto index = json::Value::object();
    index["schema"] = "gufo-prefill-telemetry-v1";
    index["binary"] = "tokens.bin";
    index["layer_records"] = "layers.jsonl";
    index["pass"] = pass_;
    index["start"] = static_cast<std::size_t>(start_);
    index["tokens"] = static_cast<std::size_t>(tokens_);
    index["header_bytes"] = 32;
    index["record_bytes"] = 36;
    index["records"] = records_;
    index["policy_identity"] = policy_;
    index["router_weight_basis"] = "original-top10-normalized";
    index["population_layers"] = 48;
    index["layers"] = json::Value::array();
    for (unsigned l = 0; l < settings_.layers.size(); ++l)
      if (Selected(l))
        index["layers"].push_back(static_cast<std::size_t>(l));
    index["layout"] =
        "little-endian <II7f: token,layer,top1_mass,top4_mass,gate_l2,gate0,"
        "gate1,gate2,gate3";
    const auto destination = directory_ / "telemetry.index.json";
    // The index is the completion marker; failed chunks leave no index.
    const auto temporary = directory_ / "index.partial";
    std::ofstream out(temporary, std::ios::binary);
    out << index.dump() << '\n';
    out.close();
    if (!out)
      throw std::runtime_error("prefill telemetry index write failed");
    std::filesystem::rename(temporary, destination);
    finished_ = true;
    return destination;
  }

private:
  void CheckFiles() const {
    if (!binary_ || !layers_)
      throw std::runtime_error("prefill telemetry file write failed");
  }
  const PrefillTelemetrySettings& settings_;
  std::uint64_t pass_, records_{0}, expected_records_{0};
  std::uint32_t start_, tokens_;
  std::string policy_;
  std::filesystem::path directory_;
  std::ofstream binary_, layers_;
  std::array<bool, 48> population_layers_{};
  bool finished_{false};
};

inline std::unique_ptr<PrefillTelemetryPass> MakePrefillTelemetryPass(
    const PrefillTelemetrySettings& settings, std::uint64_t pass,
    std::uint32_t start, std::uint32_t tokens, std::string_view policy) {
  if (!settings.enabled)
    return nullptr;
  return std::make_unique<PrefillTelemetryPass>(settings, pass, start, tokens,
                                                std::string(policy));
}

inline std::vector<TelemetryRecord> ReadTelemetryRecords(
    const std::filesystem::path& path, std::uint64_t expected_pass) {
  std::ifstream in(path, std::ios::binary);
  std::array<char, 8> magic{};
  std::array<std::uint32_t, 4> format{};
  std::uint64_t pass = 0;
  in.read(magic.data(), 8);
  in.read(reinterpret_cast<char*>(format.data()), 16);
  in.read(reinterpret_cast<char*>(&pass), 8);
  if (!in || std::string_view(magic.data(), 8) != "GUFOPFT1" ||
      format != std::array<std::uint32_t, 4>{1, 32, 36, 0} ||
      pass != expected_pass)
    throw std::runtime_error("invalid telemetry binary header");
  std::vector<TelemetryRecord> records;
  TelemetryRecord record;
  while (in.read(reinterpret_cast<char*>(&record), sizeof(record))) {
    for (float value : record.values)
      if (!std::isfinite(value))
        throw std::runtime_error("nonfinite telemetry record");
    records.push_back(record);
  }
  if (in.gcount() || !in.eof())
    throw std::runtime_error("truncated telemetry binary record");
  return records;
}

}  // namespace gufo::models::qwen38_flash_next
#endif
