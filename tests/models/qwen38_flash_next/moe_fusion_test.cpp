#include "src/models/qwen38_flash_next/moe_fusion.hpp"

#include <cstdio>
#include <stdexcept>

#include "src/models/qwen38_flash_next/device_accept.hpp"
#include "src/models/qwen38_flash_next/draft_chain.hpp"
namespace q = gufo::models::qwen38_flash_next;
void Require(bool ok, const char* message) {
  if (!ok)
    throw std::runtime_error(message);
}
int main() {
  try {
    for (unsigned rows = 1; rows <= 8; ++rows) {
      Require(q::DecodeMoeFusionShape(rows, 10, 2560, 4, false, false),
              "production decode refused");
      Require(!q::DecodeMoeFusionShape(rows, 10, 2560, 4, false, true),
              "prefill admitted");
      Require(!q::DecodeMoeFusionShape(rows, 10, 2560, 4, true, false),
              "half expert output admitted");
      const auto verify = std::uint64_t{rows} | (std::uint64_t{rows} << 16) |
                          (std::uint64_t{1} << 32);
      const auto mtp =
          q::MtpGraphKey(rows, true, true, false, true, false, false);
      const auto chain = q::DraftChainKey(rows, 4, 15, true);
      for (const auto key : {verify, mtp, chain}) {
        Require(!(key & q::kMoeFusionKey), "fusion bit collides");
        Require(((key | q::kMoeFusionKey) ^ key) == q::kMoeFusionKey,
                "fusion key changes shape fields");
      }
    }
    Require(!q::DecodeMoeFusionShape(9, 10, 2560, 4, false, false),
            "matrix rows admitted");
    Require(!q::DecodeMoeFusionShape(1, 17, 2560, 4, false, false),
            "slot order outside narrow contract");
    Require(!q::DecodeMoeFusionShape(1, 10, 2561, 4, false, false),
            "ragged shape admitted");
    Require(!q::DecodeMoeFusionShape(1, 10, 2560, 8, false, false),
            "wrong stream count admitted");
    q::DecodeCycleOptions options;
    std::string error;
    const auto parse = [&](const char* value) {
      return q::ParseDecodeCycleOptions(nullptr, nullptr, nullptr, nullptr,
                                        nullptr, &options, &error, nullptr,
                                        nullptr, value);
    };
    Require(parse(nullptr) && !options.moe_fusion, "fusion default not off");
    Require(parse("1") && options.moe_fusion, "fusion on refused");
    Require(parse("0") && !options.moe_fusion, "fusion off refused");
    Require(!parse("") && !parse("true"), "invalid fusion switch admitted");
    std::puts(
        "PASS narrow fusion eligibility, default-off parser and graph bit "
        "layouts");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
