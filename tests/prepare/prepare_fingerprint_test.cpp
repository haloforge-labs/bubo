#include <cassert>
#include <filesystem>
#include <iostream>
#include <string>

#include "src/core/prepare/fingerprint_table.hpp"
#include "src/core/prepare/metadata_json.hpp"
#include "tests/prepare/fixture_catalogue.hpp"

int main() {
  using namespace bubo::prepare;
  const auto text = metadata::Read(
      std::filesystem::u8path(BUBO_PREPARE_FIXTURES) / "fingerprints.json");
  const auto shard = fixture::kCatalogue[0];
  const auto valid = LoadFingerprintTable(text, shard);
  assert(valid && valid.table.tensors.size() == 1);
  assert(valid.table.tensors[0].offset == 0);
  assert(valid.table.tensors[0].checked_length == 3);
  assert(valid.table.tensors[0].sha256 == fixture::kAbcHash);
  // Format-only boundary cases; the same three-byte abc fixture is placed at
  // a logical absolute offset above 4 GiB. No production payload is opened.
  auto wide = shard;
  wide.size = 4294967299ULL;
  auto json = gufo::json::parse(text);
  auto tensor = json.find("tensors")->items()[0];
  tensor["offset"] = "4294967296";
  json["tensors"] = gufo::json::Value::array();
  json["tensors"].push_back(tensor);
  assert(LoadFingerprintTable(json.dump(), wide));
  for (int mutation = 0; mutation < 10; ++mutation) {
    auto broken = gufo::json::parse(text);
    auto item = broken.find("tensors")->items()[0];
    switch (mutation) {
      case 0:
        item["checked_length"] = "2";
        break;
      case 1:
        item["offset"] = "1";
        break;
      case 2:
        item["tensor_bytes"] = "0";
        break;
      case 3:
        item["sha256"] = "bad";
        break;
      case 4:
        item["offset"] = "18446744073709551616";
        break;
      case 5:
        item["offset"] = "00";
        break;
      case 6:
        item["offset"] = 0;
        break;
      case 7:
        item["tensor_name"] = "bad\nname";
        break;
      case 8:
        item["offset"] = "18446744073709551615";
        break;
      case 9:
        item["extra"] = true;
        break;
    }
    broken["tensors"] = gufo::json::Value::array();
    broken["tensors"].push_back(item);
    assert(!LoadFingerprintTable(broken.dump(), shard));
  }
  json = gufo::json::parse(text);
  json["tensors"].push_back(json.find("tensors")->items()[0]);
  assert(!LoadFingerprintTable(json.dump(), shard));
  json = gufo::json::parse(text);
  json["file_sha256"] = std::string(64, '0');
  assert(!LoadFingerprintTable(json.dump(), shard));
  json = gufo::json::parse(text);
  json["tensors"] = gufo::json::Value::array();
  assert(!LoadFingerprintTable(json.dump(), shard));
  assert(!LoadFingerprintTable(text, fixture::kCatalogue[1]));
  assert(!LoadFingerprintTable("{}", shard));
  assert(!LoadFingerprintTable(std::string(metadata::kMaxJsonBytes + 1, ' '),
                               shard));
  std::cout
      << "PASS: fingerprint fixture identity, absolute 64-bit offsets, "
         "checked-length policy, duplicate/range/overflow/schema rejection\n";
}
