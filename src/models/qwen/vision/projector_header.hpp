#ifndef GUFO_MODELS_QWEN_VISION_PROJECTOR_HEADER_HPP_
#define GUFO_MODELS_QWEN_VISION_PROJECTOR_HEADER_HPP_

// What the Qwen vision encoder accepts of an image projector ("mmproj") file,
// judged from the file header alone: metadata, tensor names, shapes and types.
// No tensor data is read, and nothing here touches the GPU.
//
// The constants below are copies of the values hard-coded in
// src/models/qwen/vision/encoder.hip (the file-scope constants at lines 27-31
// and the Encoder::Impl constructor, lines 374-476). encoder.hip is HIP code
// and cannot include this header yet. When it does, it should take these
// values from here, so that there is one place. Until then, change both when
// the projector changes.

#include <array>
#include <cstdint>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace gufo::models::qwen::vision {

inline constexpr std::string_view kProjectorArchitecture = "clip";
inline constexpr std::string_view kProjectorFileType = "mmproj";
inline constexpr std::string_view kProjectorType = "qwen3vl_merger";
inline constexpr std::uint32_t kProjectorBlocks = 27;
inline constexpr std::uint32_t kProjectorPatchSize = 16;
inline constexpr std::uint32_t kProjectorImageSize = 768;
inline constexpr std::uint32_t kProjectorSpatialMerge = 2;
inline constexpr std::uint32_t kProjectorHidden = 1152;
inline constexpr std::uint32_t kProjectorFeedForward = 4304;
inline constexpr std::uint32_t kProjectorHeads = 16;
inline constexpr std::uint32_t kProjectorMergeHidden =
    kProjectorHidden * kProjectorSpatialMerge * kProjectorSpatialMerge;
// Tensors in the projector file, as the encoder counts them.
inline constexpr std::uint64_t kProjectorTensorCount = 334;
// Text-model widths the encoder serves: Qwen3.8-27B, Flash Next,
// Qwen3.6-35B-A3B.
inline constexpr std::array<std::uint32_t, 3> kProjectorWidths = {5120, 2560,
                                                                  2048};

/// The outcome of `CheckProjectorHeader`. Each flag is false when an earlier
/// one is false: a later check is not meaningful then.
struct ProjectorHeaderCheck {
  bool kind_ok{false};    ///< A clip file of type mmproj with a vision tower.
  bool family_ok{false};  ///< The Qwen "qwen3vl_merger" projector.
  bool dims_ok{false};  ///< Sizes, normalisation and tensors the encoder needs.
};

namespace detail {
inline bool ProjectorTensorIs(const core::GgufReader& reader,
                              std::string_view name,
                              const std::vector<std::uint64_t>& shape,
                              core::GgmlType type) {
  const auto* tensor = reader.FindTensor(name);
  return tensor != nullptr && tensor->dimensions == shape &&
         tensor->type == type;
}
}  // namespace detail

/// `width` is the hidden size of the text model the projector is paired with
/// (`clip.vision.projection_dim` must equal it).
[[nodiscard]] inline ProjectorHeaderCheck CheckProjectorHeader(
    const core::GgufReader& reader, std::uint32_t width) {
  ProjectorHeaderCheck check;
  check.kind_ok =
      reader.GetMetadataString("general.architecture") ==
          kProjectorArchitecture &&
      reader.GetMetadataString("general.type") == kProjectorFileType &&
      reader.GetMetadataBool("clip.has_vision_encoder") == true;
  if (!check.kind_ok)
    return check;
  check.family_ok =
      reader.GetMetadataString("clip.projector_type") == kProjectorType;
  if (!check.family_ok)
    return check;

  bool dims = false;
  for (const auto served : kProjectorWidths)
    dims = dims || served == width;
  const auto expect = [&](std::string_view key, std::uint64_t value) {
    dims = dims && reader.GetMetadataUint64(key) == value;
  };
  expect("clip.vision.projection_dim", width);
  expect("clip.vision.patch_size", kProjectorPatchSize);
  expect("clip.vision.embedding_length", kProjectorHidden);
  expect("clip.vision.feed_forward_length", kProjectorFeedForward);
  expect("clip.vision.block_count", kProjectorBlocks);
  expect("clip.vision.attention.head_count", kProjectorHeads);
  expect("clip.vision.spatial_merge_size", kProjectorSpatialMerge);
  expect("clip.vision.image_size", kProjectorImageSize);
  dims = dims && reader.GetMetadataBool("clip.use_gelu") == true &&
         reader.GetMetadataFloat32(
             "clip.vision.attention.layer_norm_epsilon") == 1e-6F;
  for (const auto* key : {"clip.vision.image_mean", "clip.vision.image_std"}) {
    const auto* value = reader.FindMetadata(key);
    const auto* array =
        value ? std::get_if<std::vector<double>>(&value->value) : nullptr;
    dims = dims && array != nullptr &&
           *array == std::vector<double>{0.5, 0.5, 0.5};
  }
  // The merger takes the last layer only: no deepstack layer is enabled.
  const auto* deepstack =
      reader.FindMetadata("clip.vision.is_deepstack_layers");
  const auto* layers =
      deepstack ? std::get_if<std::vector<std::uint64_t>>(&deepstack->value)
                : nullptr;
  dims = dims && layers != nullptr && layers->size() == kProjectorBlocks;
  if (dims)
    for (const auto enabled : *layers)
      dims = dims && enabled == 0;
  // The tensors that depend on the text-model width, and the tensor count.
  using core::GgmlType;
  dims =
      dims && reader.GetTensorCount() == kProjectorTensorCount &&
      detail::ProjectorTensorIs(reader, "mm.0.weight",
                                {kProjectorMergeHidden, kProjectorMergeHidden},
                                GgmlType::kBF16) &&
      detail::ProjectorTensorIs(reader, "mm.0.bias", {kProjectorMergeHidden},
                                GgmlType::kF32) &&
      detail::ProjectorTensorIs(reader, "mm.2.weight",
                                {kProjectorMergeHidden, width},
                                GgmlType::kBF16) &&
      detail::ProjectorTensorIs(reader, "mm.2.bias", {width}, GgmlType::kF32);
  check.dims_ok = dims;
  return check;
}

}  // namespace gufo::models::qwen::vision

#endif  // GUFO_MODELS_QWEN_VISION_PROJECTOR_HEADER_HPP_
