#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>

#include "src/models/gemma4/vision/weights.hpp"
#include "tests/models/gemma4/check.hpp"
#include "tests/models/gemma4/gguf_builder.hpp"

namespace vision = gufo::models::gemma4::vision;
using gemma4_test::Require;

namespace {

gemma4_test::Metadata ProjectorMetadata() {
  return {{"general.architecture", std::string("clip")},
          {"general.type", std::string("mmproj")},
          {"clip.has_vision_encoder", true},
          {"clip.vision.projection_dim", std::uint32_t{2816}},
          {"clip.vision.patch_size", std::uint32_t{16}},
          {"clip.vision.embedding_length", vision::kHidden},
          {"clip.vision.feed_forward_length", vision::kFfn},
          {"clip.vision.block_count", vision::kLayers},
          {"clip.vision.attention.head_count", vision::kHeads},
          {"clip.vision.attention.layer_norm_epsilon", vision::kEps},
          {"clip.vision.image_mean", gemma4_test::F32Array{{0, 0, 0}}},
          {"clip.vision.image_std", gemma4_test::F32Array{{1, 1, 1}}}};
}

void CheckProjectorType(gemma4_test::Metadata metadata,
                        std::string_view expected_error) {
  const gemma4_test::GgufImage image(metadata);
  Require(image.reader() != nullptr, image.error());
  try {
    (void)vision::Weights::Resolve(*image.reader(), 2816);
    Require(false, "fixture without tensors was accepted");
  } catch (const std::invalid_argument& error) {
    Require(
        error.what() == expected_error,
        "unexpected projector validation error: " + std::string(error.what()));
  }
}

}  // namespace

int main() {
  return gemma4_test::Run([] {
    constexpr std::string_view kAccepted =
        "Gemma 4 vision sidecar: invalid tensor v.patch_embd.weight";
    constexpr std::string_view kRejected =
        "Gemma 4 vision sidecar: not a gemma4v projector";
    auto metadata = ProjectorMetadata();
    metadata["clip.vision.projector_type"] = std::string("gemma4v");
    CheckProjectorType(metadata, kAccepted);
    metadata.erase("clip.vision.projector_type");
    metadata["clip.projector_type"] = std::string("gemma4v");
    CheckProjectorType(metadata, kAccepted);
    metadata.erase("clip.projector_type");
    CheckProjectorType(metadata, kRejected);
    metadata["clip.projector_type"] = std::string("other");
    CheckProjectorType(metadata, kRejected);
    metadata["clip.projector_type"] = std::string("gemma4v");
    metadata["clip.vision.projector_type"] = std::string("other");
    CheckProjectorType(metadata, kRejected);
  });
}
