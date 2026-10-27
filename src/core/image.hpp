#ifndef GUFO_CORE_IMAGE_HPP_
#define GUFO_CORE_IMAGE_HPP_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace gufo::core {

inline constexpr std::size_t kMaxEncodedImageBytes = 20 * 1024 * 1024;
inline constexpr std::size_t kMaxImagePixels = 32 * 1024 * 1024;
/// Image tokens one request may carry when GUFO_MAX_IMAGE_TOKENS is unset, for
/// the engines whose admission reservation grows with it (Qwen 3.8 Flash
/// Next). The cap counts every image in the prompt, history included, so an
/// agent session that keeps screenshots needs room: 65536 is about 32 full-HD
/// screenshots (2025 tokens each) or four images of the largest accepted size
/// (16384 tokens each), and reserves 640 MiB per session.
inline constexpr std::uint32_t kDefaultRequestImageTokens = 65536;

struct Image {
  std::uint32_t width{0};
  std::uint32_t height{0};
  /// Packed, unpremultiplied RGB8, in display orientation.
  std::vector<std::uint8_t> pixels;
};

/// PNG, JPEG and WebP. Decoding checks dimensions before allocating pixels.
[[nodiscard]] Image DecodeImage(std::span<const std::uint8_t> bytes);
/// Pillow-compatible bicubic resize (Image.resize(..., BICUBIC)): Keys cubic
/// a = -0.5 with the filter widened when shrinking, 22-bit fixed-point
/// weights, horizontal pass then vertical, each rounded to 8 bits. Bit-exact
/// with Pillow and with the Gemma 4 image processor's resize.
[[nodiscard]] Image ResizeBicubic(const Image& image, std::uint32_t width,
                                  std::uint32_t height);
[[nodiscard]] std::vector<std::uint8_t> ReadImageFile(
    const std::filesystem::path& path);
/// Shared across all messages of one HTTP request.
struct ImageReadBudget {
  /// False when the loaded model takes no image input; reading is refused
  /// before any data URL is decoded or any URL is fetched.
  bool accepts_images{true};
  std::size_t remaining_bytes{kMaxEncodedImageBytes};
  std::chrono::steady_clock::time_point deadline{
      std::chrono::steady_clock::now() + std::chrono::seconds(15)};
};

/// Network-order IPv4/IPv6 address; excludes local and special-use ranges.
[[nodiscard]] bool IsPublicImageAddress(
    std::span<const std::uint8_t> address) noexcept;
/// Base64 data URLs or public HTTPS destinations. DNS results and every
/// redirect's actual socket address are checked before connecting.
[[nodiscard]] std::vector<std::uint8_t> ReadImageUrl(std::string_view url,
                                                     ImageReadBudget& budget);
[[nodiscard]] std::vector<std::uint8_t> ReadImageUrl(std::string_view url);

}  // namespace gufo::core
#endif
