#include <algorithm>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <string_view>
#include <vector>

#include "src/models/gemma4/draft_vocab_quant.hpp"
#include "src/models/gemma4/kernels/rocm/draft_upload.hpp"
#include "tests/models/gemma4/check.hpp"

namespace g4 = gufo::models::gemma4;
namespace gpu = g4::rocm;
using gemma4_test::Require;
using gufo::core::GgmlType;

namespace {

// Records the real host upload policy without allocating or calling HIP.
struct RecordingUploader {
  std::uint16_t storage{0};
  std::size_t copies{0}, conversions{0}, quantizations{0};
  std::string error;

  void Fail(const std::string& message) { error = message; }

  gpu::DeviceTensor Copy(const g4::TensorRef& t) {
    ++copies;
    return {&storage, t.type, static_cast<std::uint32_t>(t.cols),
            static_cast<std::uint32_t>(t.rows), 1};
  }

  void Half(gpu::DeviceTensor& t, std::string_view) {
    ++conversions;
    t.type = GgmlType::kF16;
  }

  gpu::DeviceTensor QuantizeBf16(const g4::TensorRef& t) {
    ++quantizations;
    return {&storage, GgmlType::kQ8_0, static_cast<std::uint32_t>(t.cols),
            static_cast<std::uint32_t>(t.rows), 1};
  }
};

void Check() {
  RecordingUploader up;
  g4::TensorRef host{&up.storage, GgmlType::kBF16, 1024, 262144, 1, 0, 0, {}};
  host.name = "token_embd.weight";
  const auto device = gpu::UploadDraftProjection(host, up);
  Require(device.type == GgmlType::kQ8_0 && up.quantizations == 1 &&
              up.copies == 0 && up.conversions == 0 && up.error.empty(),
          "BF16 vocabulary must quantize directly to the Q8_0 intermediate");
  Require(device.cols == 1024 && device.rows == 262144 &&
              host.type == GgmlType::kBF16,
          "vocabulary quantization must preserve shape and source");
  Require(gpu::ShouldRepackDraftVocabAsQ4K(device),
          "BF16 vocabulary must enter the guarded Q4_K repack");
  for (const auto name :
       {"nextn.pre_projection.weight", "nextn.post_projection.weight"}) {
    RecordingUploader projection;
    host.name = name;
    Require(
        gpu::UploadDraftProjection(host, projection).type == GgmlType::kF16 &&
            projection.copies == 1 && projection.conversions == 1 &&
            projection.quantizations == 0,
        "pre/post projections must keep their F16 conversion");
  }
  for (const auto type : {GgmlType::kQ4_0, GgmlType::kQ8_0, GgmlType::kQ4_K,
                          GgmlType::kF16, GgmlType::kF32}) {
    for (const std::uint32_t cols : {96U, 1024U}) {
      RecordingUploader native;
      g4::TensorRef tensor{&native.storage, type, cols, 3, 1, 0, 0, {}};
      tensor.name = "token_embd.weight";
      const auto result = gpu::UploadDraftProjection(tensor, native);
      Require(result.type == type && result.cols == cols && result.rows == 3 &&
                  native.copies == 1 && native.conversions == 0 &&
                  native.quantizations == 0 && native.error.empty(),
              "native heads must keep their old copy and type");
      Require(gpu::ShouldRepackDraftVocabAsQ4K(result) ==
                  (type == GgmlType::kQ8_0 && cols % 256 == 0),
              "native Q8_0 repack decision must equal the RC condition");
    }
  }
  for (const std::uint32_t cols : {0U, 32U, 96U, 255U}) {
    RecordingUploader invalid;
    g4::TensorRef tensor{
        &invalid.storage, GgmlType::kBF16, cols, 3, 1, 0, 0, {}};
    tensor.name = "token_embd.weight";
    Require(gpu::UploadDraftProjection(tensor, invalid).empty() &&
                invalid.error.find("token_embd.weight") != std::string::npos &&
                invalid.error.find("256") != std::string::npos &&
                invalid.quantizations == 0 && invalid.copies == 0,
            "BF16 vocabulary must reject invalid Q4_K alignment before upload");
  }
}

// CPU load fixture for the repacker's exact constant-negative branch. The
// policy, Q8_0 encoder, repack decision and named scale validator are real;
// only the GPU constant-block repack is replaced by its independent formula.
struct ConstantUploader : RecordingUploader {
  std::vector<gufo::quant::block_q8_0> q8;
  gufo::quant::block_q4_K q4{};
  std::size_t repacks{0}, validations{0};

  gpu::DeviceTensor QuantizeBf16(const g4::TensorRef& t) {
    ++quantizations;
    q8.resize(t.rows * t.cols / 32);
    if (!g4::QuantizeDraftVocabQ8_0(t.data, t.rows, t.cols, q8, &error)) {
      return {};
    }
    return {q8.data(), GgmlType::kQ8_0, static_cast<std::uint32_t>(t.cols),
            static_cast<std::uint32_t>(t.rows), 1};
  }

  gpu::DeviceTensor Load(const g4::TensorRef& t) {
    auto head = gpu::UploadDraftProjection(t, *this);
    if (!gpu::ShouldRepackDraftVocabAsQ4K(head)) {
      return head;
    }
    ++repacks;
    Require(
        q8.size() == 8 &&
            std::all_of(q8.begin(), q8.end(),
                        [&](const auto& b) {
                          return b.d == q8[0].d &&
                                 std::all_of(std::begin(b.qs), std::end(b.qs),
                                             [](auto q) { return q == -127; });
                        }),
        "constant-negative load fixture must have eight identical Q8 blocks");
    const float minimum = gufo::quant::Fp16ToFloat(q8[0].d) * 127.0F;
    q4.dmin =
        std::bit_cast<std::uint16_t>(static_cast<_Float16>(minimum / 63.0F));
    ++validations;
    if (!g4::ValidateDraftVocabQ4KScales(std::span(&q4, 1), &error, t.name)) {
      return {};
    }
    return {&q4, GgmlType::kQ4_K, head.cols, head.rows, 1};
  }
};

void CheckDefaultGuard() {
  for (const std::uint16_t bits : {0xca7b, 0xca80}) {
    std::vector<std::uint16_t> values(256, bits);
    ConstantUploader up;
    g4::TensorRef host{values.data(), GgmlType::kBF16, 256, 1, 1, 0, 0, {}};
    host.name = "token_embd.weight";
    const auto head = up.Load(host);
    Require(up.quantizations == 1 && up.repacks == 1 && up.validations == 1 &&
                up.copies == 0 && up.conversions == 0,
            "default BF16 load must quantize, repack and validate Q4_K");
    if (bits == 0xca80) {
      Require(
          head.empty() &&
              up.error.find("token_embd.weight") != std::string::npos &&
              up.error.find("nonfinite dmin") != std::string::npos,
          "default load must reject overflowing Q4_K scales with tensor name");
    } else {
      Require(!head.empty() && head.type == GgmlType::kQ4_K && up.error.empty(),
              "default load must accept a wide but safe Q4_K head");
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  return gemma4_test::Run([&] {
    if (argc == 2 && std::string_view(argv[1]) == "--default-guard") {
      CheckDefaultGuard();
    } else {
      Require(argc == 1, "unexpected vocabulary policy test argument");
      Check();
    }
  });
}
