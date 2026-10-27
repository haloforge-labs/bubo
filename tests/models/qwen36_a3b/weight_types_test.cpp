#include <array>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen36_a3b/engine.hpp"
#include "src/models/qwen36_a3b/weight_format.hpp"

namespace gufo::models::qwen36_a3b {

// Exercise the real Get gate with a host mapping standing in for an uploaded
// region. No stream, device allocation, upload or other HIP API is invoked.
struct EngineTestAccess {
  static bool Get(std::unique_ptr<core::GgufReader> reader, Tensor* out,
                  std::string* error, std::uint32_t columns = 0) {
    Engine engine;
    engine.gguf_ = std::move(reader);
    if (columns != 0) {
      auto* tensor = const_cast<core::GgufTensorInfo*>(
          engine.gguf_->FindTensor("test.weight"));
      tensor->dimensions[0] = columns;
    }
    const auto region = engine.gguf_->GetMappedRegions().front();
    engine.regions_.push_back(const_cast<void*>(region.data));
    const bool ok = engine.Get("test.weight", out, error);
    engine.regions_.clear();  // borrowed host storage, never hipFree it
    return ok;
  }
};

}  // namespace gufo::models::qwen36_a3b

namespace {

using gufo::core::GgmlType;
namespace a3b = gufo::models::qwen36_a3b;

void Require(bool ok, const char* message) {
  if (!ok)
    throw std::runtime_error(message);
}

template<class T>
void Append(std::vector<std::uint8_t>& bytes, T value) {
  const auto offset = bytes.size();
  bytes.resize(offset + sizeof(value));
  std::memcpy(bytes.data() + offset, &value, sizeof(value));
}

std::vector<std::uint8_t> Fixture(GgmlType type, std::uint32_t columns,
                                  bool expert) {
  std::vector<std::uint8_t> bytes{'G', 'G', 'U', 'F'};
  Append<std::uint32_t>(bytes, 3);
  Append<std::uint64_t>(bytes, 1);  // tensors
  Append<std::uint64_t>(bytes, 0);  // metadata
  constexpr std::string_view name = "test.weight";
  Append<std::uint64_t>(bytes, name.size());
  bytes.insert(bytes.end(), name.begin(), name.end());
  Append<std::uint32_t>(bytes, expert ? 3 : 2);
  Append<std::uint64_t>(bytes, columns);
  Append<std::uint64_t>(bytes, 3);
  if (expert)
    Append<std::uint64_t>(bytes, 5);
  Append<std::uint32_t>(bytes, static_cast<std::uint32_t>(type));
  Append<std::uint64_t>(bytes, 0);
  bytes.resize((bytes.size() + 31) / 32 * 32);
  bytes.resize(bytes.size() + gufo::quant::EncodedSizeBytes(
                                  type, columns * 3 * (expert ? 5 : 1)));
  return bytes;
}

void Check(GgmlType type, std::uint32_t columns, bool expert, bool accepted,
           std::uint32_t override_columns = 0) {
  const auto bytes = Fixture(type, columns, expert);
  std::string error;
  auto reader =
      gufo::core::GgufReader::OpenMemory(bytes.data(), bytes.size(), &error);
  Require(reader != nullptr, error.c_str());
  const auto* expected = reader->FindTensor("test.weight")->data;
  a3b::Tensor result;
  const bool ok = a3b::EngineTestAccess::Get(std::move(reader), &result, &error,
                                             override_columns);
  Require(ok == accepted, "Get type/alignment acceptance mismatch");
  if (ok) {
    Require(error.empty(), "successful Get reported an error");
    Require(result.type == type && result.data == expected,
            "Get changed the encoded type or mapping");
    Require(result.cols == columns && result.rows == 3 &&
                result.experts == (expert ? 5U : 1U),
            "Get changed tensor dimensions");
  } else if (override_columns != 0) {
    Require(
        error == "K-quant columns must be a multiple of 256 for test.weight",
        "Get failed to reject partial K blocks");
  } else {
    Require(error == "unsupported type " +
                         std::string(gufo::core::ToString(type)) +
                         " for test.weight (this build handles F32/Q8_0/BF16)",
            "unsupported type error changed");
  }
}

}  // namespace

int main() {
  try {
    for (auto type :
         {GgmlType::kQ3_K, GgmlType::kQ4_K, GgmlType::kQ5_K, GgmlType::kQ6_K}) {
      for (auto columns : {256U, 512U, 2048U})
        for (bool expert : {false, true})
          Check(type, columns, expert, true);
      // The reader already rejects partial rows. Mutate only the synthetic
      // descriptor after parsing to exercise Get's own independent guard.
      for (auto columns : {1U, 128U, 255U, 257U, 2047U})
        Check(type, 2048, true, false, columns);
    }
    for (auto type : {GgmlType::kF32, GgmlType::kBF16, GgmlType::kQ8_0})
      for (bool expert : {false, true})
        Check(type, 288, expert, true);  // no new 256-column gate
    for (auto type :
         {GgmlType::kF16, GgmlType::kQ4_0, GgmlType::kQ4_1, GgmlType::kQ5_0,
          GgmlType::kQ5_1, GgmlType::kQ8_1, GgmlType::kQ2_K, GgmlType::kQ8_K,
          GgmlType::kIQ2_XXS, GgmlType::kIQ3_S, GgmlType::kIQ4_XS,
          GgmlType::kIQ4_NL, GgmlType::kI32})
      Check(type, 2048, true, false);
    const std::array<std::size_t, 4> row_bytes{880, 1152, 1408, 1680};
    std::size_t i = 0;
    for (auto type :
         {GgmlType::kQ3_K, GgmlType::kQ4_K, GgmlType::kQ5_K, GgmlType::kQ6_K}) {
      Require(a3b::IsKQuant(type), "K-quant dispatch gate missed a type");
      Require(gufo::quant::EncodedSizeBytes(type, 2048) == row_bytes[i++],
              "incorrect encoded head row stride");
    }
    Require(!a3b::IsKQuant(GgmlType::kQ8_0) &&
                !a3b::IsKQuant(GgmlType::kBF16) &&
                !a3b::IsKQuant(GgmlType::kF32),
            "existing types entered K-quant dispatch");
    std::cout << "PASS: A3B Get types, alignment, mapping and K row bytes\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
