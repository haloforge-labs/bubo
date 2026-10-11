#include "src/core/prepare/fingerprint_table.hpp"

#include <algorithm>
#include <set>
#include <stdexcept>

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
    Keys(root, {"schema_version", "logical_name", "file_sha256", "tensors"});
    if (Integer(Required(root, "schema_version")) != 1)
      throw std::invalid_argument("unsupported fingerprint schema");
    result.table.logical_name = String(Required(root, "logical_name"));
    result.table.file_sha256 = String(Required(root, "file_sha256"));
    if (result.table.logical_name != shard.logical_name ||
        !IsLowerHex(result.table.file_sha256, 64) ||
        result.table.file_sha256 != shard.sha256)
      throw std::invalid_argument("fingerprint table shard identity mismatch");
    const auto& tensors = Required(root, "tensors");
    if (!tensors.is_array() || tensors.empty() || tensors.size() > 4096)
      throw std::invalid_argument("expected 1..4096 tensor fingerprints");
    std::set<std::string> names;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    for (const auto& value : tensors.items()) {
      Keys(value, {"tensor_name", "offset", "tensor_bytes", "checked_length",
                   "sha256"});
      TensorFingerprint tensor{
          String(Required(value, "tensor_name")),
          Decimal(String(Required(value, "offset"))),
          Decimal(String(Required(value, "tensor_bytes"))),
          Decimal(String(Required(value, "checked_length"))),
          String(Required(value, "sha256"))};
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
}  // namespace bubo::prepare
