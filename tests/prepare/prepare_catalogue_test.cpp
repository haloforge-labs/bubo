#include <cassert>
#include <iostream>
#include <string>

#include "src/core/prepare/release_catalogue.hpp"

int main() {
  using namespace bubo::prepare;
  const auto& catalogue = ModelCatalogue();
  assert(ValidateFields(catalogue, false).empty());
  assert(!ValidateFields(catalogue, true).empty());
  // Independent contract values: addendum section 8 / recipe sections 1-3.
  std::uint64_t shards = 0, all = 0;
  std::size_t shard_count = 0;
  for (const auto& entry : catalogue) {
    all += entry.size;
    if (entry.kind == Kind::MainShard) {
      shards += entry.size;
      ++shard_count;
      assert(entry.fingerprint_table_slot.starts_with("fingerprints/main_"));
    } else {
      assert(entry.fingerprint_table_slot.empty());
    }
  }
  assert(shard_count == 4 && catalogue.size() == 8);
  assert(shards == 111334655392ULL);
  assert(all == 116617136608ULL);
  assert(PIN_SLOT_UNSLOTH_COMMIT == "766911a6b7369840a91dbcd95f9f997acaab6cd6" &&
         PIN_SLOT_BUBO_COMMIT.empty());
  assert(BUBO_HF_REPO_ID == "PIN_SLOT_BUBO_REPO_ID");
  assert(catalogue[5].file_name == "mmproj-BF16.gguf");
  assert(catalogue[5].destination ==
         "mmproj/mmproj-Qwen3.8-Flash-Next-BF16.gguf");
  for (auto path : {"../a", "/a", "C:/a", "a\\b", "a//b", "a/./b", "a/../b",
                    "a.", "a ", "CON.txt", "a/NUL", "LPT1.gguf", "a:b", "a?b"})
    assert(!IsRelativePath(path));
  for (auto path : {"a/b.gguf", "mtp/model.gguf", "名字/weight.gguf"})
    assert(IsRelativePath(path));
  for (int mutation = 0; mutation < 12; ++mutation) {
    auto copy = catalogue;
    switch (mutation) {
      case 0:
        copy[0].sha256 = "bad";
        break;
      case 1:
        copy[0].size = 0;
        break;
      case 2:
        copy[0].destination = "../outside";
        break;
      case 3:
        copy[0].licence_id = "MIT";
        break;
      case 4:
        copy[0].source_repo = "https://huggingface.co/repo";
        break;
      case 5:
        copy[0].commit = "main";
        break;
      case 6:
        copy[1].destination = copy[0].destination;
        break;
      case 7:
        copy[0].file_name = "other.gguf";
        break;
      case 8:
        copy[0].kind = static_cast<Kind>(99);
        break;
      case 9:
        copy[0].fingerprint_table_slot = "";
        break;
      case 10:
        copy[4].fingerprint_table_slot = "fingerprints/sidecar.json";
        break;
      case 11:
        copy[1].destination =
            "qwen3.8-flash-next-ud-q4_k_xl-00001-of-00004.gguf";
        break;
    }
    assert(!ValidateFields(copy, false).empty());
  }
  std::cout << "PASS: eight catalogue entries; shards=111334655392 B; "
               "all=116617136608 B; fields and fingerprint slots\n";
}
