#ifndef GUFO_MODELS_DIFFUSION_GEMMA_TOKENIZER_HPP_
#define GUFO_MODELS_DIFFUSION_GEMMA_TOKENIZER_HPP_

#include <array>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace gufo::models::diffusion_gemma {

/// Gemma 4 tokenizer from tokenizer.json: added tokens matched first
/// (longest), spaces normalised to U+2581, one BPE word per segment with
/// byte fallback, merges applied lowest rank first, leftmost on ties. This is
/// the Hugging Face `tokenizers` BPE model with this file's settings.
class Tokenizer {
public:
  explicit Tokenizer(const std::filesystem::path& root);

  [[nodiscard]] std::vector<std::int32_t> Encode(std::string_view text) const;
  /// Concatenated token text. Special (added) tokens are dropped when
  /// `skip_special` is set; byte-fallback tokens are fused into raw bytes.
  [[nodiscard]] std::string Decode(std::span<const std::int32_t> ids,
                                   bool skip_special) const;
  [[nodiscard]] bool IsSpecial(std::int32_t id) const;
  [[nodiscard]] std::int32_t TokenId(std::string_view token) const;
  [[nodiscard]] std::size_t vocab_size() const { return pieces_.size(); }

private:
  void EncodeSegment(std::string_view text,
                     std::vector<std::int32_t>* out) const;

  std::vector<std::string> pieces_;
  std::unordered_map<std::string, std::int32_t> vocab_;
  /// (left id << 32 | right id) -> (rank << 32 | merged id).
  std::unordered_map<std::uint64_t, std::uint64_t> merges_;
  std::vector<std::pair<std::string, std::int32_t>> added_;
  std::vector<bool> special_;
  std::array<std::int32_t, 256> byte_ids_{};
  /// -1 for a token that is not a byte-fallback token.
  std::vector<std::int16_t> byte_value_;
};

}  // namespace gufo::models::diffusion_gemma

#endif  // GUFO_MODELS_DIFFUSION_GEMMA_TOKENIZER_HPP_
