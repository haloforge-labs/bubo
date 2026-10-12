#include "src/core/prepare/gguf_fingerprint.hpp"

#include <algorithm>
#include <array>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>
#include <vector>

#include "src/core/crypto/sha256.hpp"
#include "src/core/quant/ggml_dequant.hpp"

namespace bubo::prepare {
namespace {
constexpr std::uint64_t kHeaderLimit = 64ULL * 1024 * 1024;
class Reader {
public:
  explicit Reader(const std::filesystem::path& path)
      : file(path, std::ios::binary) {
    if (!file)
      throw std::runtime_error("cannot open GGUF");
  }
  void Read(std::span<std::uint8_t> bytes) {
    if (bytes.size() > kHeaderLimit - position)
      throw std::runtime_error("GGUF header exceeds 64 MiB");
    file.read(reinterpret_cast<char*>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
    if (file.gcount() != static_cast<std::streamsize>(bytes.size()))
      throw std::runtime_error("truncated GGUF header");
    hash.Update(bytes);
    position += bytes.size();
  }
  std::uint64_t Number(std::size_t width) {
    std::array<std::uint8_t, 8> data{};
    Read(std::span(data.data(), width));
    std::uint64_t result = 0;
    for (std::size_t i = 0; i < width; ++i)
      result |= std::uint64_t(data[i]) << (8 * i);
    return result;
  }
  std::string String(std::uint64_t limit = 1024 * 1024) {
    const auto length = Number(8);
    if (length > limit || length > kHeaderLimit - position)
      throw std::runtime_error("oversized GGUF string");
    std::string result(static_cast<std::size_t>(length), '\0');
    Read({reinterpret_cast<std::uint8_t*>(result.data()), result.size()});
    return result;
  }
  std::uint64_t Value(std::uint64_t type, bool array_item = false) {
    constexpr std::size_t sizes[] = {1, 1, 2, 2, 4, 4, 4, 1, 0, 0, 8, 8, 8};
    if (type > 12)
      throw std::runtime_error("unknown GGUF metadata type");
    if (type == 8) {
      String();
      return 0;
    }
    if (type == 9) {
      if (array_item)
        throw std::runtime_error("nested GGUF array");
      const auto element = Number(4), count = Number(8);
      if (element > 12 || element == 9 || count > kHeaderLimit - position)
        throw std::runtime_error("invalid/oversized GGUF array");
      for (std::uint64_t i = 0; i < count; ++i)
        Value(element, true);
      return 0;
    }
    const auto result = Number(sizes[type]);
    if (type == 7 && result > 1)
      throw std::runtime_error("invalid GGUF boolean");
    return result;
  }
  std::ifstream file;
  gufo::crypto::Sha256Hasher hash;
  std::uint64_t position = 0;
};

FingerprintTable ReadTable(Reader& reader, const Entry& shard) {
  if (reader.Number(4) != 0x46554747 || reader.Number(4) != 3)
    throw std::runtime_error("expected GGUF version 3");
  const auto tensors = reader.Number(8), kv = reader.Number(8);
  if (tensors > 4096 || kv > 65536)
    throw std::runtime_error("invalid GGUF header counts");
  std::uint64_t alignment = 32;
  std::set<std::string> keys;
  for (std::uint64_t i = 0; i < kv; ++i) {
    const auto key = reader.String(65535);
    const auto type = reader.Number(4);
    if (key.empty() || !keys.insert(key).second)
      throw std::runtime_error("empty/duplicate GGUF metadata key");
    const auto value = reader.Value(type);
    if (key == "general.alignment") {
      if (type != 4)
        throw std::runtime_error("GGUF alignment must be uint32");
      alignment = value;
    }
  }
  if (!alignment || alignment > 1024 * 1024 || (alignment & (alignment - 1)))
    throw std::runtime_error("invalid GGUF alignment");
  FingerprintTable table;
  table.schema_version = 2;
  table.generator_version = kFingerprintGeneratorVersion;
  table.logical_name = shard.logical_name;
  table.file_sha256 = shard.sha256;
  std::set<std::string> names;
  for (std::uint64_t i = 0; i < tensors; ++i) {
    TensorFingerprint tensor{};
    tensor.tensor_name = reader.String(1024);
    if (tensor.tensor_name.empty() ||
        !names.insert(tensor.tensor_name).second ||
        std::ranges::any_of(tensor.tensor_name,
                            [](unsigned char c) { return c < 32 || c == 127; }))
      throw std::runtime_error("empty/duplicate/invalid GGUF tensor name");
    const auto dimensions = reader.Number(4);
    if (!dimensions || dimensions > 8)
      throw std::runtime_error("invalid GGUF dimension count");
    std::uint64_t elements = 1;
    for (std::uint64_t dim = 0; dim < dimensions; ++dim) {
      const auto value = reader.Number(8);
      if (!value || elements > UINT64_MAX / value)
        throw std::runtime_error("GGUF shape overflow");
      elements *= value;
      tensor.dimensions.push_back(value);
    }
    const auto raw_type = reader.Number(4);
    if (raw_type > UINT16_MAX)
      throw std::runtime_error("unsupported GGUF tensor type");
    tensor.type = static_cast<std::uint32_t>(raw_type);
    const auto type = static_cast<gufo::core::GgmlType>(raw_type);
    const auto block = gufo::quant::QuantizedBlockElements(type);
    tensor.tensor_bytes =
        gufo::quant::EncodedSizeBytes(type, static_cast<std::size_t>(elements));
    if (!tensor.tensor_bytes || (block && tensor.dimensions.front() % block))
      throw std::runtime_error(
          "unsupported GGUF storage type or misaligned shape");
    tensor.relative_offset = reader.Number(8);
    if (tensor.relative_offset % alignment)
      throw std::runtime_error("unaligned GGUF tensor offset");
    tensor.checked_length = std::min(kFingerprintBytes, tensor.tensor_bytes);
    table.tensors.push_back(std::move(tensor));
  }
  const auto padding = (alignment - reader.position % alignment) % alignment;
  std::vector<std::uint8_t> pad(static_cast<std::size_t>(padding));
  reader.Read(pad);
  table.header_bytes = reader.position;
  table.header_sha256 = reader.hash.FinishHex();
  if (table.tensors.empty() && table.header_bytes != shard.size)
    throw std::runtime_error(
        "metadata-only GGUF contains unpinned trailing bytes");
  std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
  for (auto& tensor : table.tensors) {
    if (table.header_bytes > shard.size ||
        tensor.relative_offset > shard.size - table.header_bytes)
      throw std::runtime_error("GGUF tensor offset past file");
    tensor.offset = table.header_bytes + tensor.relative_offset;
    if (tensor.tensor_bytes > shard.size - tensor.offset)
      throw std::runtime_error("GGUF tensor payload truncated");
    ranges.emplace_back(tensor.offset, tensor.offset + tensor.tensor_bytes);
  }
  std::ranges::sort(ranges);
  for (std::size_t i = 1; i < ranges.size(); ++i)
    if (ranges[i].first < ranges[i - 1].second)
      throw std::runtime_error("overlapping GGUF tensors");
  return table;
}

void HashPrefixes(Reader& reader, FingerprintTable& table) {
  std::vector<std::uint8_t> buffer(kFingerprintBytes);
  for (auto& tensor : table.tensors) {
    reader.file.seekg(static_cast<std::streamoff>(tensor.offset),
                      std::ios::beg);
    reader.file.read(reinterpret_cast<char*>(buffer.data()),
                     static_cast<std::streamsize>(tensor.checked_length));
    if (reader.file.gcount() !=
        static_cast<std::streamsize>(tensor.checked_length))
      throw std::runtime_error("short tensor prefix: " + tensor.tensor_name);
    tensor.sha256 = gufo::crypto::Sha256Hex(
        {buffer.data(), static_cast<std::size_t>(tensor.checked_length)});
  }
}
}  // namespace

FingerprintResult GenerateGgufFingerprint(const std::filesystem::path& path,
                                          const Entry& shard) {
  FingerprintResult result;
  try {
    if (shard.kind != Kind::MainShard ||
        std::filesystem::file_size(path) != shard.size)
      throw std::runtime_error("GGUF shard size/identity mismatch");
    Reader reader(path);
    result.table = ReadTable(reader, shard);
    HashPrefixes(reader, result.table);
  } catch (const std::exception& error) {
    result.table = {};
    result.error = error.what();
  }
  return result;
}

std::string CheckGgufFingerprint(const std::filesystem::path& path,
                                 const Entry& shard,
                                 const FingerprintTable& pin,
                                 bool full_verify) {
  try {
    if (shard.kind != Kind::MainShard ||
        std::filesystem::file_size(path) != shard.size)
      return "GGUF shard size mismatch";
    if (pin.schema_version != 2 ||
        pin.generator_version != kFingerprintGeneratorVersion ||
        pin.logical_name != shard.logical_name ||
        pin.file_sha256 != shard.sha256)
      return "missing complete GGUF fingerprint pin";
    if (full_verify)
      return gufo::crypto::Sha256FileHex(path) == shard.sha256
                 ? ""
                 : "GGUF full SHA-256 mismatch";
    Reader reader(path);
    auto actual = ReadTable(reader, shard);
    if (actual.header_bytes != pin.header_bytes ||
        actual.header_sha256 != pin.header_sha256 ||
        actual.tensors.size() != pin.tensors.size())
      return "GGUF header/tensor table mismatch";
    for (std::size_t i = 0; i < actual.tensors.size(); ++i) {
      const auto& a = actual.tensors[i];
      const auto& p = pin.tensors[i];
      if (a.tensor_name != p.tensor_name || a.type != p.type ||
          a.dimensions != p.dimensions || a.offset != p.offset ||
          a.relative_offset != p.relative_offset ||
          a.tensor_bytes != p.tensor_bytes ||
          a.checked_length != p.checked_length)
        return "GGUF tensor table mismatch: " + a.tensor_name;
    }
    HashPrefixes(reader, actual);
    for (std::size_t i = 0; i < actual.tensors.size(); ++i)
      if (actual.tensors[i].sha256 != pin.tensors[i].sha256)
        return "GGUF tensor fingerprint mismatch: " +
               actual.tensors[i].tensor_name;
    return {};
  } catch (const std::exception& error) {
    return error.what();
  }
}
std::string CheckGgufHeader(const std::filesystem::path& path,
                            const Entry& shard, const FingerprintTable& pin) {
  try {
    if (std::filesystem::file_size(path) != shard.size)
      return "GGUF shard size mismatch";
    if (pin.schema_version != 2 ||
        pin.generator_version != kFingerprintGeneratorVersion ||
        pin.logical_name != shard.logical_name ||
        pin.file_sha256 != shard.sha256)
      return "missing complete GGUF header pin";
    Reader reader(path);
    const auto actual = ReadTable(reader, shard);
    if (actual.header_bytes != pin.header_bytes ||
        actual.header_sha256 != pin.header_sha256 ||
        actual.tensors.size() != pin.tensors.size())
      return "GGUF header/tensor table mismatch";
    for (std::size_t i = 0; i < actual.tensors.size(); ++i) {
      const auto& a = actual.tensors[i];
      const auto& p = pin.tensors[i];
      if (a.tensor_name != p.tensor_name || a.type != p.type ||
          a.dimensions != p.dimensions || a.offset != p.offset ||
          a.relative_offset != p.relative_offset ||
          a.tensor_bytes != p.tensor_bytes || a.checked_length != p.checked_length)
        return "GGUF tensor table mismatch: " + a.tensor_name;
    }
    return {};
  } catch (const std::exception& error) {
    return error.what();
  }
}
}  // namespace bubo::prepare
