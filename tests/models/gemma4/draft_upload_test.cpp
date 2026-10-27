#include "src/models/gemma4/kernels/rocm/draft_upload.hpp"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <string>
#include <string_view>

#include "src/core/quant/ggml_dequant.hpp"
#include "tests/models/gemma4/check.hpp"

namespace g4 = gufo::models::gemma4;
namespace gpu = g4::rocm;
using gemma4_test::Require;
using gufo::core::GgmlType;

namespace {

// No HIP calls or payload reads: records the real upload policy's requests.
struct RecordingUploader {
  std::uint16_t storage{0};
  std::size_t copies{0}, conversions{0}, quantizations{0}, count{0}, bytes{0};
  std::string name;
  bool fail_copy{false};

  void Fail(const std::string& message) { Require(false, message); }

  gpu::DeviceTensor QuantizeBf16(const g4::TensorRef& t) {
    ++quantizations;
    bytes += gufo::quant::EncodedSizeBytes(GgmlType::kQ8_0,
                                           std::size_t{t.rows} * t.cols);
    return {&storage, GgmlType::kQ8_0, static_cast<std::uint32_t>(t.cols),
            static_cast<std::uint32_t>(t.rows), 1};
  }

  gpu::DeviceTensor Copy(const g4::TensorRef& t) {
    ++copies;
    bytes += t.SizeBytes();
    if (fail_copy) {
      return {};
    }
    return {&storage, t.type, static_cast<std::uint32_t>(t.cols),
            static_cast<std::uint32_t>(t.rows),
            static_cast<std::uint32_t>(t.experts)};
  }

  void Half(gpu::DeviceTensor& t, std::string_view tensor_name) {
    Require(!t.empty() && t.type == GgmlType::kBF16,
            "only a successfully copied BF16 projection may be converted");
    ++conversions;
    count += std::size_t{t.rows} * t.cols * t.experts;
    name = tensor_name;
    t.type = GgmlType::kF16;
  }
};

void CheckProjection(std::uint32_t cols, std::uint32_t rows,
                     std::string_view name) {
  for (const auto type : {GgmlType::kBF16, GgmlType::kQ4_0, GgmlType::kQ8_0,
                          GgmlType::kF16, GgmlType::kF32}) {
    RecordingUploader up;
    g4::TensorRef host;
    host.data = &up.storage;
    host.type = type;
    host.cols = cols;
    host.rows = rows;
    host.name = name;
    const auto device = gpu::UploadDraftProjection(host, up);
    const bool bf16 = type == GgmlType::kBF16;
    const bool quantized = bf16 && name == "token_embd.weight";
    Require(
        up.copies == (quantized ? 0U : 1U) &&
            up.quantizations == (quantized ? 1U : 0U) &&
            up.bytes == (quantized
                             ? gufo::quant::EncodedSizeBytes(
                                   GgmlType::kQ8_0, std::size_t{rows} * cols)
                             : host.SizeBytes()),
        "default BF16 vocabulary must quantize; other projections copy once");
    Require(device.data == &up.storage && device.cols == cols &&
                device.rows == rows && device.experts == 1,
            "conversion must preserve the allocation and shape");
    Require(
        device.type == (quantized ? GgmlType::kQ8_0
                        : bf16    ? GgmlType::kF16
                                  : type),
        "default vocabulary uses Q8_0 for Q4_K repack; BF16 pre/post use F16");
    Require(up.conversions == (bf16 && !quantized ? 1U : 0U),
            "quantized and F16 heads must keep the copy-only path");
    Require(up.count == (bf16 && !quantized ? std::size_t{cols} * rows : 0),
            "conversion must account for every value, including the vocab");
    Require(up.name == (bf16 && !quantized ? std::string(name) : std::string()),
            "overflow diagnostics must retain the source tensor name");
    Require(host.type == type && host.data == &up.storage,
            "the source tensor must remain unchanged");
    if (bf16 && !quantized) {
      Require(up.bytes == up.count * sizeof(std::uint16_t),
              "BF16 to binary16 must keep the resident payload size");
      up.fail_copy = true;
      Require(
          gpu::UploadDraftProjection(host, up).empty() && up.conversions == 1,
          "a failed copy must not queue a conversion");
    }
  }
}

}  // namespace

int main() {
  return gemma4_test::Run([] {
    for (const std::uint32_t target : {2816U, 5376U}) {
      CheckProjection(2 * target, 1024, "nextn.pre_projection.weight");
      CheckProjection(1024, target, "nextn.post_projection.weight");
    }
    CheckProjection(1024, 262144, "token_embd.weight");
  });
}
