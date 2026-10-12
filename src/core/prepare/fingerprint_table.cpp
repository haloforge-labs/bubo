#include "src/core/prepare/fingerprint_table.hpp"

#include <algorithm>
#include <set>
#include <stdexcept>

#include "src/core/prepare/fingerprints/embedded_tables.hpp"
#include "src/core/prepare/metadata_json.hpp"

namespace bubo::prepare {
FingerprintResult LoadFingerprintTable(std::string_view text,
                                       const Entry& shard) {
  FingerprintResult result;
  try {
    using namespace metadata;
    if (shard.kind != Kind::MainShard)
      throw std::invalid_argument("fingerprint table requires a main shard");
    const auto root = Parse(text);
    const auto schema = Integer(Required(root, "schema_version"));
    Keys(root, {"schema_version", "logical_name", "file_sha256", "tensors",
                "generator_version", "header_bytes", "header_sha256"});
    if (schema != 1 && schema != 2)
      throw std::invalid_argument("unsupported fingerprint schema");
    result.table.schema_version = static_cast<std::uint32_t>(schema);
    if (schema == 2) {
      result.table.generator_version =
          String(Required(root, "generator_version"));
      result.table.header_bytes =
          Decimal(String(Required(root, "header_bytes")));
      result.table.header_sha256 = String(Required(root, "header_sha256"));
      if (result.table.generator_version != "bubo-gguf-fingerprint-v2" ||
          result.table.header_bytes < 24 ||
          result.table.header_bytes >
              std::min<std::uint64_t>(shard.size, 64ULL * 1024 * 1024) ||
          !IsLowerHex(result.table.header_sha256, 64))
        throw std::invalid_argument("invalid complete header pin");
    } else {
      Keys(root, {"schema_version", "logical_name", "file_sha256", "tensors"});
    }
    result.table.logical_name = String(Required(root, "logical_name"));
    result.table.file_sha256 = String(Required(root, "file_sha256"));
    if (result.table.logical_name != shard.logical_name ||
        !IsLowerHex(result.table.file_sha256, 64) ||
        result.table.file_sha256 != shard.sha256)
      throw std::invalid_argument("fingerprint table shard identity mismatch");
    const auto& tensors = Required(root, "tensors");
    if (!tensors.is_array() || tensors.size() > 4096 ||
        (tensors.empty() &&
         (schema != 2 || result.table.header_bytes != shard.size)))
      throw std::invalid_argument(
          "expected complete tensor fingerprints or metadata-only shard");
    std::set<std::string> names;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    for (const auto& value : tensors.items()) {
      Keys(value, {"tensor_name", "offset", "tensor_bytes", "checked_length",
                   "sha256", "type", "dimensions", "relative_offset"});
      TensorFingerprint tensor{
          String(Required(value, "tensor_name")),
          Decimal(String(Required(value, "offset"))),
          Decimal(String(Required(value, "tensor_bytes"))),
          Decimal(String(Required(value, "checked_length"))),
          String(Required(value, "sha256")),
          0,
          {},
          0};
      if (schema == 2) {
        const auto type = Integer(Required(value, "type"));
        if (type > UINT32_MAX)
          throw std::invalid_argument("invalid tensor type");
        tensor.type = static_cast<std::uint32_t>(type);
        tensor.relative_offset =
            Decimal(String(Required(value, "relative_offset")));
        const auto& dimensions = Required(value, "dimensions");
        if (!dimensions.is_array() || dimensions.empty() ||
            dimensions.size() > 8)
          throw std::invalid_argument("invalid tensor dimensions");
        for (const auto& dim : dimensions.items()) {
          const auto number = Decimal(String(dim));
          if (!number)
            throw std::invalid_argument("zero tensor dimension");
          tensor.dimensions.push_back(number);
        }
        if (tensor.offset < result.table.header_bytes ||
            tensor.offset - result.table.header_bytes != tensor.relative_offset)
          throw std::invalid_argument(
              "invalid absolute/relative tensor offset");
      } else {
        Keys(value, {"tensor_name", "offset", "tensor_bytes", "checked_length",
                     "sha256"});
      }
      if (tensor.tensor_name.empty() || tensor.tensor_name.size() > 1024 ||
          std::ranges::any_of(
              tensor.tensor_name,
              [](unsigned char c) { return c < 32 || c == 127; }) ||
          !names.insert(tensor.tensor_name).second)
        throw std::invalid_argument("invalid/duplicate tensor name");
      if (!tensor.tensor_bytes || tensor.offset > shard.size ||
          tensor.tensor_bytes > shard.size - tensor.offset ||
          tensor.checked_length !=
              std::min(kFingerprintBytes, tensor.tensor_bytes) ||
          !IsLowerHex(tensor.sha256, 64))
        throw std::invalid_argument("invalid tensor range/length/hash");
      ranges.emplace_back(tensor.offset, tensor.offset + tensor.tensor_bytes);
      result.table.tensors.push_back(std::move(tensor));
    }
    std::ranges::sort(ranges);
    for (std::size_t i = 1; i < ranges.size(); ++i)
      if (ranges[i].first < ranges[i - 1].second)
        throw std::invalid_argument("overlapping tensor ranges");
  } catch (const std::exception& error) {
    result.table = {};
    result.error = error.what();
  }
  return result;
}

std::string SerializeFingerprintTable(const FingerprintTable& table) {
  using gufo::json::Value;
  Value root = Value::object();
  root["schema_version"] =
      static_cast<unsigned long long>(table.schema_version);
  root["logical_name"] = table.logical_name;
  root["file_sha256"] = table.file_sha256;
  root["generator_version"] = table.generator_version;
  root["header_bytes"] = std::to_string(table.header_bytes);
  root["header_sha256"] = table.header_sha256;
  root["tensors"] = Value::array();
  for (const auto& tensor : table.tensors) {
    Value item = Value::object();
    item["tensor_name"] = tensor.tensor_name;
    item["offset"] = std::to_string(tensor.offset);
    item["tensor_bytes"] = std::to_string(tensor.tensor_bytes);
    item["checked_length"] = std::to_string(tensor.checked_length);
    item["sha256"] = tensor.sha256;
    item["type"] = static_cast<unsigned long long>(tensor.type);
    item["relative_offset"] = std::to_string(tensor.relative_offset);
    item["dimensions"] = Value::array();
    for (auto dim : tensor.dimensions)
      item["dimensions"].push_back(std::to_string(dim));
    root["tensors"].push_back(item);
  }
  return root.dump();
}

FingerprintResult ProductionFingerprintTable(const Entry& shard) {
  for (const auto& [name, text] : kEmbeddedFingerprintTables)
    if (name == shard.logical_name)
      return LoadFingerprintTable(text, shard);
  return {{}, "missing production fingerprint table"};
}
}  // namespace bubo::prepare
