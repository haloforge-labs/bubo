#include <cassert>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>

#include "src/core/crypto/sha256.hpp"
#include "src/core/prepare/gguf_fingerprint.hpp"

namespace {
void Number(std::vector<std::uint8_t>& bytes, std::uint64_t value,
            std::size_t width) {
  for (std::size_t i = 0; i < width; ++i)
    bytes.push_back(static_cast<std::uint8_t>(value >> (8 * i)));
}
void String(std::vector<std::uint8_t>& bytes, std::string_view value) {
  Number(bytes, value.size(), 8);
  bytes.insert(bytes.end(), value.begin(), value.end());
}
void Write(const std::filesystem::path& path,
           const std::vector<std::uint8_t>& bytes) {
  std::ofstream file(path, std::ios::binary | std::ios::trunc);
  file.write(reinterpret_cast<const char*>(bytes.data()),
             static_cast<std::streamsize>(bytes.size()));
  assert(file.good());
}
}  // namespace
int main() {
  using namespace bubo::prepare;
  const auto directory =
      std::filesystem::temp_directory_path() /
      ("bubo-fingerprint-test-" +
       std::to_string(
           std::chrono::steady_clock::now().time_since_epoch().count()));
  std::filesystem::create_directory(directory);
  const auto path = directory / "fixture.gguf";
  std::vector<std::uint8_t> bytes;
  Number(bytes, 0x46554747, 4);
  Number(bytes, 3, 4);
  Number(bytes, 2, 8);
  Number(bytes, 1, 8);
  String(bytes, "general.name");
  Number(bytes, 8, 4);
  String(bytes, "synthetic-v1");
  const std::uint64_t tensor_bytes = kFingerprintBytes + 32;
  const std::uint64_t first_tensor_bytes = 32;
  for (int i = 0; i < 2; ++i) {
    String(bytes, i == 0 ? "tensor.a" : "tensor.b");
    Number(bytes, 1, 4);
    Number(bytes, (i == 0 ? first_tensor_bytes : tensor_bytes) / 4, 8);
    Number(bytes, 0, 4);
    Number(bytes, i * first_tensor_bytes, 8);
  }
  while (bytes.size() % 32)
    bytes.push_back(0);
  const auto start = bytes.size();
  bytes.resize(start + first_tensor_bytes + tensor_bytes, 0x41);
  std::fill(bytes.begin() + start + first_tensor_bytes, bytes.end(), 0x42);
  const auto hash = gufo::crypto::Sha256Hex(bytes);
  Entry entry{"fixture",
              "fixture.gguf",
              bytes.size(),
              hash,
              "fixture/test",
              "0123456789012345678901234567890123456789",
              "fixture.gguf",
              Kind::MainShard,
              "qwen-community-1.0",
              "fixture.gguf",
              "fingerprints/fixture.json"};
  Write(path, bytes);
  const auto generated = GenerateGgufFingerprint(path, entry);
  assert(generated && generated.table.tensors.size() == 2);
  assert(generated.table.tensors[0].checked_length == first_tensor_bytes);
  assert(generated.table.tensors[1].checked_length == kFingerprintBytes);
  const auto pin =
      LoadFingerprintTable(SerializeFingerprintTable(generated.table), entry);
  assert(pin);
  assert(CheckGgufFingerprint(path, entry, pin.table).empty());
  assert(CheckGgufFingerprint(path, entry, pin.table, true).empty());
  // Variant preserves size, header and every descriptor. The final tensor's
  // first MiB has one planted wrong byte; checking only the first tensor fails.
  auto changed = bytes;
  changed[start + first_tensor_bytes + kFingerprintBytes - 1] ^= 1;
  Write(path, changed);
  assert(!CheckGgufFingerprint(path, entry, pin.table).empty());
  auto broken_first_tensor_only = pin.table;
  broken_first_tensor_only.tensors.pop_back();
  assert(
      gufo::crypto::Sha256Hex({changed.data() + start,
                               static_cast<std::size_t>(first_tensor_bytes)}) ==
      pin.table.tensors[0].sha256);
  assert(!CheckGgufFingerprint(path, entry, broken_first_tensor_only).empty());
  std::cout << "PASS: planted same-layout variant rejected; broken "
               "first-tensor-only check accepts wrong variant\n";
  changed = bytes;
  changed.resize(changed.size() - 1);
  Write(path, changed);
  assert(!CheckGgufFingerprint(path, entry, pin.table).empty());
  changed = bytes;
  changed[56] ^= 1;
  Write(path, changed);
  assert(!CheckGgufFingerprint(path, entry, pin.table).empty());
  auto changed_pin = pin.table;
  ++changed_pin.tensors[1].dimensions[0];
  Write(path, bytes);
  assert(!CheckGgufFingerprint(path, entry, changed_pin).empty());
  changed_pin = pin.table;
  changed_pin.schema_version = 1;
  assert(!CheckGgufFingerprint(path, entry, changed_pin).empty());
  // Default deliberately checks prefixes only; --verify covers arbitrary tails.
  changed = bytes;
  changed.back() ^= 1;
  Write(path, changed);
  assert(CheckGgufFingerprint(path, entry, pin.table).empty());
  assert(!CheckGgufFingerprint(path, entry, pin.table, true).empty());
  changed = bytes;
  for (std::size_t i = 8; i < 16; ++i)
    changed[i] = 0xff;
  Write(path, changed);
  assert(!GenerateGgufFingerprint(path, entry));
  // The production first shard is metadata only: zero tensors is accepted
  // exclusively when the complete file is covered by the pinned header SHA.
  std::vector<std::uint8_t> metadata_only;
  Number(metadata_only, 0x46554747, 4);
  Number(metadata_only, 3, 4);
  Number(metadata_only, 0, 8);
  Number(metadata_only, 0, 8);
  metadata_only.resize(32, 0);
  const auto metadata_hash = gufo::crypto::Sha256Hex(metadata_only);
  auto metadata_entry = entry;
  metadata_entry.size = metadata_only.size();
  metadata_entry.sha256 = metadata_hash;
  Write(path, metadata_only);
  const auto metadata_pin = GenerateGgufFingerprint(path, metadata_entry);
  assert(metadata_pin && metadata_pin.table.tensors.empty());
  assert(LoadFingerprintTable(SerializeFingerprintTable(metadata_pin.table),
                              metadata_entry));
  assert(
      CheckGgufFingerprint(path, metadata_entry, metadata_pin.table).empty());
  metadata_only.back() = 1;
  Write(path, metadata_only);
  assert(
      !CheckGgufFingerprint(path, metadata_entry, metadata_pin.table).empty());
  metadata_only.push_back(0);
  metadata_entry.size = metadata_only.size();
  Write(path, metadata_only);
  assert(!GenerateGgufFingerprint(path, metadata_entry));
  // All production tables have exact pinned identity and complete schema.
  for (const auto& model : ModelCatalogue()) {
    if (model.kind != Kind::MainShard)
      continue;
    const auto production = ProductionFingerprintTable(model);
    assert(production && production.table.schema_version == 2);
    assert(!production.table.tensors.empty() ||
           production.table.header_bytes == model.size);
  }
  std::filesystem::remove_all(directory);
  std::cout << "PASS: every tensor, exact size, GGUF header/table/shape, "
               "truncation, bounded counts, schema1 refusal, full-SHA tail, "
               "production pin loading\n";
}
