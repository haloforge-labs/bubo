#include "src/models/gemma4/draft_vocab_quant.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "src/core/quant/ggml_dequant.hpp"
#include "tests/models/gemma4/check.hpp"

namespace g4 = gufo::models::gemma4;
using gemma4_test::Require;
using gufo::quant::block_q8_0;

namespace {

float Decode(std::uint16_t value) {
  return std::bit_cast<float>(std::uint32_t{value} << 16);
}

// Exhaustive nearest finite binary16, ties to even: independent of the
// production cast, including subnormal scale rounding.
std::uint16_t ReferenceHalf(float value) {
  std::uint16_t best = 0;
  double distance = std::numeric_limits<double>::infinity();
  for (std::uint16_t h = 0; h <= 0x7bff; ++h) {
    const double diff = std::abs(double(value) - gufo::quant::Fp16ToFloat(h));
    if (diff < distance || (diff == distance && (h & 1) == 0)) {
      best = h;
      distance = diff;
    }
  }
  return best;
}

void CheckMatrix(std::size_t cols) {
  constexpr std::size_t rows = 3;
  std::vector<std::uint16_t> source(rows * cols);
  for (std::size_t i = 32; i < source.size(); ++i) {
    const float value = float(int((i * 73) % 511) - 255) / 32.0F;
    source[i] =
        static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(value) >> 16);
  }
  // One outlier amidst tiny values, mixed signs, ties away from zero, and
  // tiny scales (including a binary16 scale rounded all the way to zero).
  source[37] = 0x4980;  // 2^20, much larger than binary16 weights can hold.
  for (std::size_t i = 64; i < 96; ++i) {
    const float v = i == 64 ? 127.0F : (i & 1 ? -1.5F : 0.5F);
    source[i] =
        static_cast<std::uint16_t>(std::bit_cast<std::uint32_t>(v) >> 16);
  }
  for (std::size_t i = 96; i < 128; ++i) {
    source[i] = (i & 1) ? 0xb580 : 0x3580;  // +/- 2^-20.
  }
  std::vector<block_q8_0> actual(source.size() / 32);
  std::string error;
  Require(g4::QuantizeDraftVocabQ8_0(source.data(), rows, cols, actual, &error),
          "quantize fixed BF16 matrix: " + error);
  std::vector<std::uint8_t> reference(actual.size() * 34, 0);
  bool saw_scale_rounding = false;
  for (std::size_t b = 0; b < actual.size(); ++b) {
    float amax = 0;
    for (std::size_t i = 0; i < 32; ++i) {
      amax = std::max(amax, std::abs(Decode(source[b * 32 + i])));
    }
    const float d = amax / 127.0F;
    const float inverse = d == 0 ? 0 : 1.0F / d;
    const auto half = ReferenceHalf(d);
    const double stored = gufo::quant::Fp16ToFloat(half);
    std::memcpy(reference.data() + b * 34, &half, 2);
    saw_scale_rounding |= stored != d;
    for (std::size_t i = 0; i < 32; ++i) {
      const float value = Decode(source[b * 32 + i]);
      const int q = static_cast<int>(std::round(value * inverse));
      reference[b * 34 + 2 + i] =
          static_cast<std::uint8_t>(static_cast<std::int8_t>(q));
      // Half a quantization step, plus the stored scale's rounding error
      // accumulated over at most 127 levels, plus FP32 reciprocal error.
      const double bound = 0.5 * d + 127 * std::abs(double(d) - stored) +
                           4 * std::numeric_limits<float>::epsilon() * amax;
      Require(std::abs(value - stored * actual[b].qs[i]) <= bound,
              "Q8_0 error exceeded the quantization and scale-rounding bound");
    }
  }
  Require(saw_scale_rounding, "fixed data must exercise scale rounding");
  Require(std::memcmp(actual.data(), reference.data(), reference.size()) == 0,
          "exact GGML Q8_0 layout, half scale rounding and signed codes");
  Require(actual[0].d == 0 &&
              std::all_of(std::begin(actual[0].qs), std::end(actual[0].qs),
                          [](auto v) { return v == 0; }),
          "zero block must encode as 34 zero bytes");
  Require(actual[2].qs[1] == -2 && actual[2].qs[2] == 1,
          "Q8_0 codes round halfway values away from zero");
  // Chunked production uploads must give the same bytes as whole matrices.
  std::vector<block_q8_0> chunks(actual.size());
  for (std::size_t r = 0; r < rows; ++r) {
    Require(g4::QuantizeDraftVocabQ8_0(
                source.data() + r * cols, 1, cols,
                std::span(chunks).subspan(r * cols / 32, cols / 32), &error),
            "quantize row chunk");
  }
  Require(std::memcmp(actual.data(), chunks.data(), reference.size()) == 0,
          "chunk boundaries must preserve the row-major encoding");
}

void CheckInvalid() {
  std::vector<std::uint16_t> values(256, 0);
  std::vector<block_q8_0> output(8);
  std::string error;
  for (const auto cols : {0U, 31U, 33U}) {
    Require(
        !g4::QuantizeDraftVocabQ8_0(values.data(), 1, cols, output, &error) &&
            !error.empty(),
        "reject invalid Q8_0 row alignment");
  }
  Require(!g4::QuantizeDraftVocabQ8_0(values.data(), 1, 256,
                                      std::span(output).first(7), &error),
          "reject wrong output size");
  Require(!g4::QuantizeDraftVocabQ8_0(values.data(), 0, 256, output, &error),
          "reject an empty matrix");
  Require(!g4::QuantizeDraftVocabQ8_0(nullptr, 1, 256, output, &error),
          "reject a null source");
  for (const std::uint16_t invalid : {0x7f80, 0xff80, 0x7fc1, 0x7f7f}) {
    values[15] = invalid;
    error.clear();
    Require(
        !g4::QuantizeDraftVocabQ8_0(values.data(), 1, 256, output, &error) &&
            !error.empty(),
        "nonfinite weights or overflowing scales must fail");
  }
  values.assign(256,
                1);  // Smallest positive BF16: reciprocal scale overflows F32.
  Require(
      g4::QuantizeDraftVocabQ8_0(values.data(), 1, 256, output, &error) &&
          output[0].d == 0 && output[0].qs[0] == 127,
      "tiny finite BF16 values must quantize without an infinite reciprocal");
  Require(!g4::QuantizeDraftVocabQ8_0(values.data(),
                                      std::numeric_limits<std::size_t>::max(),
                                      256, output, &error),
          "matrix size overflow must fail before reading the source");
}

void CheckQ4KScales() {
  // Independent reproduction of the repacker's constant-negative branch:
  // fitted scale = 0, fitted minimum = -x, stored dmin = half(minimum / 63).
  const auto constant_q4 = [](const block_q8_0& q8) {
    Require(std::all_of(std::begin(q8.qs), std::end(q8.qs),
                        [](auto q) { return q == -127; }),
            "constant negative BF16 block must quantize to -127");
    gufo::quant::block_q4_K q4{};
    const float minimum = gufo::quant::Fp16ToFloat(q8.d) * 127.0F;
    q4.dmin =
        std::bit_cast<std::uint16_t>(static_cast<_Float16>(minimum / 63.0F));
    return q4;
  };
  std::vector<std::uint16_t> source(256, 0xca80);  // -4194304, finite BF16.
  std::vector<block_q8_0> q8(8);
  std::string error;
  Require(g4::QuantizeDraftVocabQ8_0(source.data(), 1, 256, q8, &error) &&
              gufo::quant::Fp16ToFloat(q8[0].d) == 33024.0F,
          "wide constant must remain accepted by the unchanged Q8_0 route");
  std::vector<gufo::quant::block_q4_K> q4{constant_q4(q8[0])};
  Require(q4[0].dmin == 0x7c00,
          "review reproduction must overflow Q4_K dmin to positive infinity");
  Require(!g4::ValidateDraftVocabQ4KScales(q4, &error),
          "Q4_K must reject nonfinite dmin from BF16 0xca80 (-4194304)");
  Require(error.find("Q4_K") != std::string::npos &&
              error.find("nonfinite dmin") != std::string::npos &&
              error.find("super-block 0") != std::string::npos,
          "Q4_K rejection must identify the nonfinite scale and super-block");

  // Largest constant BF16 magnitude whose Q8_0-rounded minimum / 63 is
  // finite. The next BF16 magnitude (0xca7c) rounds its Q8 scale to 32512.
  source.assign(256, 0xca7b);  // -4112384, well outside binary16 weights.
  Require(g4::QuantizeDraftVocabQ8_0(source.data(), 1, 256, q8, &error) &&
              gufo::quant::Fp16ToFloat(q8[0].d) == 32384.0F,
          "largest safe constant BF16 must quantize without changing Q8_0");
  q4[0] = constant_q4(q8[0]);
  Require(g4::ValidateDraftVocabQ4KScales(q4, &error),
          "wide but safe constant BF16 must pass the Q4_K scale scan");
  source.assign(256, 0xca7c);
  Require(g4::QuantizeDraftVocabQ8_0(source.data(), 1, 256, q8, &error),
          "next constant BF16 must still be valid Q8_0");
  q4[0] = constant_q4(q8[0]);
  Require(!g4::ValidateDraftVocabQ4KScales(q4, &error),
          "first unsafe constant BF16 must fail the Q4_K scale scan");

  // At the exact Q8-scale boundary: 127*d8/63 < 65520, the FP16 overflow
  // midpoint. 32496 is the largest half d8 below that bound; 32512 is next.
  q8[0].d = std::bit_cast<std::uint16_t>(static_cast<_Float16>(32496.0F));
  q4[0] = constant_q4(q8[0]);
  Require(q4[0].dmin == 0x7bff && g4::ValidateDraftVocabQ4KScales(q4, &error),
          "Q4_K must accept the largest safe Q8_0 constant-block scale");
  ++q8[0].d;
  q4[0] = constant_q4(q8[0]);
  Require(q4[0].dmin == 0x7c00 && !g4::ValidateDraftVocabQ4KScales(q4, &error),
          "next Q8_0 scale must overflow Q4_K dmin and fail");

  q4.assign(3, {});
  q4[0].d = 0x3c00;
  q4[0].dmin = 0x4000;  // Their bitwise OR looks nonfinite; each is finite.
  q4[1].d = q4[1].dmin = 0x7bff;
  Require(g4::ValidateDraftVocabQ4KScales(q4, &error),
          "scan must accept zero and maximum finite d and dmin");
  for (const std::uint16_t invalid : {0x7c00, 0xfc00, 0x7e00, 0x7fff, 0xffff}) {
    for (auto member :
         {&gufo::quant::block_q4_K::d, &gufo::quant::block_q4_K::dmin}) {
      q4[2].*member = invalid;
      error.clear();
      Require(!g4::ValidateDraftVocabQ4KScales(q4, &error) &&
                  error.find("super-block 2") != std::string::npos,
              "scan every block and reject either scale's signed Inf/NaN");
      q4[2].*member = 0;
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  return gemma4_test::Run([&] {
    if (argc == 2 && std::string_view(argv[1]) == "--q4-k-range") {
      CheckQ4KScales();
      return;
    }
    Require(argc == 1, "unexpected quantizer test argument");
    CheckMatrix(1024);  // Real vocabulary width, aligned to 256.
    CheckMatrix(96);    // Q8_0 row width aligned only to 32.
    CheckInvalid();
  });
}
