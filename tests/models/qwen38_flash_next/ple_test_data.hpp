#ifndef GUFO_TESTS_QWEN38_FLASH_NEXT_PLE_TEST_DATA_HPP_
#define GUFO_TESTS_QWEN38_FLASH_NEXT_PLE_TEST_DATA_HPP_

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <vector>

#include "src/core/platform/fs.hpp"
#include "src/models/qwen/tokenizer.hpp"
#include "src/models/qwen38_flash_next/ngram.hpp"

namespace gufo::models::qwen38_flash_next::test {
inline void Require(bool ok, const char* message) {
  if (!ok)
    throw std::runtime_error(message);
}

struct RealPleTable {
  std::unique_ptr<core::GgufReader> header;
  std::unique_ptr<NgramTable> table;
  Config config;
  std::uint32_t vocab{0};  ///< Tokenizer size; the header-only config has none.
  std::vector<std::int32_t> prompt;
};

/// Reads metadata and selected table rows only; never binds or loads a model.
inline RealPleTable OpenRealPle(const std::filesystem::path& first) {
  RealPleTable real;
  std::string error;
  real.header = core::GgufReader::OpenHeaderFile(first, &error);
  Require(real.header != nullptr, error.c_str());
  auto config = Config::FromGguf(*real.header, true, &error);
  Require(config.has_value(), error.c_str());
  real.config = *config;
  auto tokenizer =
      tokenization::QwenTokenizer::CreateFromGguf(*real.header, &error);
  Require(tokenizer != nullptr, error.c_str());
  real.vocab = static_cast<std::uint32_t>(tokenizer->GetVocabSize());
  Require(real.vocab > 0, "tokenizer has no vocabulary");
  const auto tokens = tokenizer->Encode(
      "Write exactly the integers 1 through 30, separated by spaces.");
  real.prompt.assign(tokens.begin(), tokens.end());
  Require(real.prompt.size() >= 9, "real prompt has too few tokens");
  std::vector<std::filesystem::path> paths;
  for (const auto& entry :
       std::filesystem::directory_iterator(first.parent_path()))
    if (entry.is_regular_file() && entry.path().extension() == ".gguf")
      paths.push_back(entry.path());
  std::ranges::sort(paths);
  for (const auto& path : paths) {
    auto shard = core::GgufReader::OpenHeaderFile(path, &error);
    Require(shard != nullptr, error.c_str());
    const auto* tensor = shard->FindTensor("per_layer_token_embd.weight");
    if (!tensor)
      continue;
    Require(tensor->dimensions.size() == 2 && tensor->dimensions[0] == 160,
            "unsupported real PLE geometry");
    const auto offset = static_cast<const std::uint8_t*>(tensor->data) -
                        static_cast<const std::uint8_t*>(shard->GetData());
    const auto file = platform::OpenReadFile(path);
    real.table = NgramTable::Open(file, offset, tensor->dimensions[1], 160,
                                  tensor->type, &error);
    platform::CloseFile(file);
    Require(real.table != nullptr, error.c_str());
    const auto type = core::ToString(tensor->type);
    std::printf(
        "table=%s type=%.*s rows=%llu row_bytes=%zu prompt_tokens=%zu\n",
        path.string().c_str(), static_cast<int>(type.size()), type.data(),
        static_cast<unsigned long long>(real.table->Rows()),
        real.table->RowBytes(), real.prompt.size());
    break;
  }
  Require(real.table != nullptr, "real production PLE table not found");
  return real;
}
}  // namespace gufo::models::qwen38_flash_next::test
#endif  // GUFO_TESTS_QWEN38_FLASH_NEXT_PLE_TEST_DATA_HPP_
