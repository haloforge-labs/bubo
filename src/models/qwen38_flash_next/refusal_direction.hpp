#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_REFUSAL_DIRECTION_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_REFUSAL_DIRECTION_HPP_

#include <array>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <string_view>
#include <utility>

#include "src/core/crypto/sha256.hpp"
#include "src/core/gguf_reader.hpp"

namespace gufo::models::qwen38_flash_next {

inline constexpr std::uint64_t kRefusalDirectionKey = std::uint64_t{1} << 52;
inline constexpr unsigned kRefusalHidden = 2560;
inline constexpr unsigned kRefusalThreads = 256;
// Compatibility for version-1 direction files made before strength metadata:
// median of the 96 per-site through-origin fits in the original derivation.
inline constexpr float kRefusalFittedStrength = 1.955837064F;

enum class RefusalMode : std::uint8_t { kSites, kStream };

struct RefusalControls {
  bool enabled{false};
  std::optional<float> strength_override;
  RefusalMode mode{RefusalMode::kSites};
  unsigned first_layer{4};
  unsigned last_layer{44};
};

// Explicit CLI settings override the matching process environment values.
struct RefusalOverrides {
  std::optional<std::string> file;
  std::optional<std::string> enabled;
  std::optional<std::string> strength;
  std::optional<std::string> mode;
  std::optional<std::string> first_layer;
  std::optional<std::string> last_layer;
};

// The package marker and other side files live beside the model shards.
inline constexpr const char* kPackagedRefusalDirection =
    "refusal/refusal-r-d180ec15-fitted.gguf";

inline bool ResolvePackagedRefusalDirection(
    const std::filesystem::path& package_root, std::filesystem::path* path,
    std::string* error) {
  *path = package_root / kPackagedRefusalDirection;
  std::error_code ec;
  if (!package_root.empty() && std::filesystem::is_regular_file(*path, ec))
    return true;
  if (error)
    *error =
        "Refusal direction: packaged default missing: " + path->string() +
        "; install the direction file there or set --refusal-direction-file";
  return false;
}

inline bool ParseRefusalControls(
    const char* enabled_text, const char* strength_text, const char* mode_text,
    const char* first_text, const char* last_text, bool direction_path_present,
    RefusalControls* controls, std::string* error) {
  const auto fail = [&](std::string message) {
    if (error)
      *error = "Refusal direction: " + std::move(message);
    return false;
  };
  if (!controls)
    return fail("missing settings output");
  RefusalControls parsed;
  // A supplied file was the original opt-in. Keep that debug workflow while
  // the catalogue's explicit off value takes precedence when present.
  if (enabled_text == nullptr || *enabled_text == '\0') {
    parsed.enabled = direction_path_present;
  } else if (std::string_view(enabled_text) == "on") {
    parsed.enabled = true;
  } else if (std::string_view(enabled_text) == "off") {
    parsed.enabled = false;
  } else {
    return fail("GUFO_REFUSAL_DIRECTION_ENABLED must be on or off");
  }
  if (strength_text != nullptr && *strength_text != '\0') {
    char* end = nullptr;
    const float value = std::strtof(strength_text, &end);
    if (end == strength_text || *end != '\0' || !std::isfinite(value) ||
        value < 0.0F || value > 4.0F) {
      return fail(
          "GUFO_REFUSAL_DIRECTION_STRENGTH must be finite and in [0, 4]");
    }
    parsed.strength_override = value;
  }
  if (mode_text != nullptr && *mode_text != '\0') {
    if (std::string_view(mode_text) == "sites") {
      parsed.mode = RefusalMode::kSites;
    } else if (std::string_view(mode_text) == "stream") {
      parsed.mode = RefusalMode::kStream;
    } else {
      return fail("GUFO_REFUSAL_DIRECTION_MODE must be sites or stream");
    }
  }
  const auto parse_layer = [&](const char* text, unsigned fallback,
                               unsigned* value, const char* name) {
    if (text == nullptr || *text == '\0') {
      *value = fallback;
      return true;
    }
    unsigned parsed_value = 0;
    const auto end = text + std::strlen(text);
    const auto result = std::from_chars(text, end, parsed_value);
    if (result.ec != std::errc{} || result.ptr != end || parsed_value == 0 ||
        parsed_value > 47) {
      if (error)
        *error = std::string("Refusal direction: ") + name +
                 " must be an integer from 1 to 47";
      return false;
    }
    *value = parsed_value;
    return true;
  };
  if (!parse_layer(first_text, 4, &parsed.first_layer,
                   "GUFO_REFUSAL_DIRECTION_LAYER_START") ||
      !parse_layer(last_text, 44, &parsed.last_layer,
                   "GUFO_REFUSAL_DIRECTION_LAYER_END"))
    return false;
  if (parsed.first_layer == 0 || parsed.last_layer == 0 ||
      parsed.first_layer > parsed.last_layer) {
    return fail("layer range must satisfy 1 <= start <= end <= 47");
  }
  *controls = parsed;
  return true;
}

inline std::set<std::string, std::less<>> RefusalSites() {
  std::set<std::string, std::less<>> sites;
  for (unsigned i = 0; i < 48; ++i) {
    const auto prefix = "blk." + std::to_string(i) + ".";
    sites.insert(prefix +
                 (i % 4 == 3 ? "attn_output.weight" : "ssm_out.weight"));
    sites.insert(prefix + "ffn_down_shexp.weight");
  }
  return sites;
}

struct RefusalDirection {
  std::array<float, kRefusalHidden> r{};
  std::set<std::string, std::less<>> sites;
  float fitted_strength{kRefusalFittedStrength};
  std::string identity;
};

inline std::string RefusalConfigurationIdentity(
    const RefusalDirection& direction, RefusalMode mode, float strength,
    unsigned first, unsigned last) {
  return direction.identity +
         "mode=" + (mode == RefusalMode::kSites ? "sites" : "stream") +
         "\nstrength_bits=" +
         std::to_string(std::bit_cast<std::uint32_t>(strength)) +
         "\nlayers=" + std::to_string(first) + ":" + std::to_string(last) +
         "\n";
}

inline std::shared_ptr<const RefusalDirection> LoadRefusalDirection(
    const std::filesystem::path& path, std::string* error) {
  const auto fail =
      [&](std::string message) -> std::shared_ptr<const RefusalDirection> {
    if (error)
      *error = "Refusal direction: " + message;
    return {};
  };
  if (path.empty())
    return fail("a nonempty file path is required");
  std::error_code ec;
  const auto size = std::filesystem::file_size(path, ec);
  if (ec || size > 65536)
    return fail("cannot read file, or file exceeds 64 KiB");
  auto reader = core::GgufReader::OpenHeaderFile(path, error);
  if (!reader)
    return {};
  if (reader->GetMetadataString("general.architecture") !=
          "refusal-direction" ||
      reader->GetMetadataString("refusal.version") != "1")
    return fail("expected refusal-direction format version 1");
  for (const auto& tensor : reader->GetTensors())
    if (tensor.name != "direction.r")
      return fail("unknown tensor name: " + std::string(tensor.name));
  const auto* tensor = reader->FindTensor("direction.r");
  if (!tensor || reader->GetTensorCount() != 1 ||
      tensor->type != core::GgmlType::kF32 ||
      tensor->dimensions != std::vector<std::uint64_t>{kRefusalHidden} ||
      tensor->size_bytes != kRefusalHidden * sizeof(float))
    return fail("direction.r must be one F32 vector of length 2560");
  auto out = std::make_shared<RefusalDirection>();
  const auto fitted = reader->GetMetadataString("refusal.fitted_strength");
  if (reader->FindMetadata("refusal.fitted_strength") && !fitted)
    return fail("refusal.fitted_strength must be a string");
  if (fitted) {
    const std::string fitted_text(*fitted);
    char* end = nullptr;
    const float value = std::strtof(fitted_text.c_str(), &end);
    if (end == fitted_text.c_str() || *end != '\0' || !std::isfinite(value) ||
        value < 0.0F || value > 4.0F)
      return fail("refusal.fitted_strength must be finite and in [0, 4]");
    out->fitted_strength = value;
  }
  std::memcpy(out->r.data(), tensor->data, sizeof(out->r));
  double norm = 0;
  for (float v : out->r) {
    if (!std::isfinite(v))
      return fail("direction.r contains a nonfinite value");
    norm += double(v) * v;
  }
  if (std::abs(norm - 1) > 1e-5)
    return fail("direction.r must have unit norm");
  auto list = reader->GetMetadataString("refusal.sites");
  if (!list)
    return fail("missing site list refusal.sites");
  std::string_view remaining = *list;
  while (!remaining.empty()) {
    const auto end = remaining.find('\n');
    const auto name = remaining.substr(0, end);
    if (!out->sites.emplace(name).second)
      return fail("duplicate site: " + std::string(name));
    if (end == std::string_view::npos)
      break;
    remaining.remove_prefix(end + 1);
    if (remaining.empty())
      return fail("empty site at end of list");
  }
  const auto expected = RefusalSites();
  for (const auto& site : out->sites)
    if (!expected.contains(site))
      return fail("unknown site: " + site);
  for (const auto& site : expected)
    if (!out->sites.contains(site))
      return fail("missing site: " + site);
  out->identity = "refusal_projection=v1:" +
                  gufo::crypto::Sha256Hex(
                      {reinterpret_cast<const std::uint8_t*>(out->r.data()),
                       sizeof(out->r)}) +
                  "\n";
  return out;
}

}  // namespace gufo::models::qwen38_flash_next
#endif
