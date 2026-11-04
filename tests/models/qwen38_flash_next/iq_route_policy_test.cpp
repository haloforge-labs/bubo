#include "src/models/qwen38_flash_next/kernels/rocm/iq_route_policy.hpp"

#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace q = gufo::models::qwen38_flash_next::rocm;

void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

int main() {
  try {
    Require(q::kIqRouteMinRows == 256, "default threshold changed");
    Require(!q::UseIqWmmaRoute(false, 255, 256), "off below threshold");
    Require(!q::UseIqWmmaRoute(false, 256, 256), "off at threshold");
    Require(!q::UseIqWmmaRoute(true, 255, 256), "IQ below threshold");
    Require(q::UseIqWmmaRoute(true, 256, 256), "IQ at threshold");
    Require(q::UseIqWmmaRoute(true, 257, 256), "IQ above threshold");
    Require(!q::UseIqWmmaRoute(true, 511, 512), "override below threshold");
    Require(q::UseIqWmmaRoute(true, 512, 512), "override at threshold");
    Require(q::UseIqWmmaRoute(true, 1, 0), "zero selects IQ");
    Require(!q::UseIqWmmaRoute(true, 1024, 65536),
            "large threshold selects MMQ");

    // A restore remainder of 34 rows in a 5602-row prompt takes the route the
    // cold passes of that prompt took; decode and short prompts keep theirs.
    Require(q::IqRouteRows(true, 34, 5602) == 5602, "prefill counts the prompt");
    Require(q::UseIqWmmaRoute(true, q::IqRouteRows(true, 34, 5602), 256),
            "restore remainder takes the WMMA route of its prompt");
    Require(!q::UseIqWmmaRoute(true, q::IqRouteRows(true, 34, 0), 256),
            "no prompt size: the pass counts its own rows");
    Require(!q::UseIqWmmaRoute(true, q::IqRouteRows(true, 34, 100), 256),
            "a short prompt stays on MMQ in every pass");
    Require(q::IqRouteRows(true, 4096, 100) == 4096, "never fewer than the pass");
    Require(q::IqRouteRows(false, 1, 5602) == 1, "decode counts its own rows");
    Require(!q::UseIqWmmaRoute(true, q::IqRouteRows(false, 4, 5602), 256),
            "a decode or verify pass does not inherit the prompt route");

    Require(!q::UseIqWmmaRoute(true, q::IqRouteRows(true, 34, 255), 256) &&
                q::UseIqWmmaRoute(true, q::IqRouteRows(true, 34, 256), 256),
            "the prompt length decides at the threshold");
    Require(q::IqRouteRows(true, 300, 300) == 300, "equal rows");
    Require(!q::UseIqWmmaRoute(false, q::IqRouteRows(true, 34, 5602), 256),
            "route off stays off whatever the prompt length");
    std::uint32_t rows = 999;
    Require(q::ParseIqRouteMinRows("256", &rows) && rows == 256,
            "parse 256");
    Require(q::ParseIqRouteMinRows("0", &rows) && rows == 0,
            "parse zero");
    Require(q::ParseIqRouteMinRows("4294967295", &rows) &&
                rows == std::numeric_limits<std::uint32_t>::max(),
            "parse maximum");
    for (std::string_view bad : {"", "garbage", "256x", "-1", "+1", " 256",
                                 "4294967296"}) {
      rows = 999;
      Require(!q::ParseIqRouteMinRows(bad, &rows) && rows == 999,
              "garbage accepted or output changed");
    }
    Require(!q::ParseIqRouteMinRows("1", nullptr), "null output accepted");

    // GUFO_ROUTED_WMMA_IQ: 0 is the MMQ fallback, 1 the route without the
    // pair kernel, 2 and unset the pair-kernel route; anything else warns and
    // keeps the default, 2.
    using Mode = q::IqWmmaMode;
    const auto unset = q::ParseIqWmmaMode(nullptr);
    Require(unset.mode == Mode::kPair2 && !unset.unsupported,
            "unset is the pair route");
    Require(q::ParseIqWmmaMode("0").mode == Mode::kOff &&
                !q::ParseIqWmmaMode("0").unsupported,
            "0 is off");
    Require(q::ParseIqWmmaMode("1").mode == Mode::kOn &&
                !q::ParseIqWmmaMode("1").unsupported,
            "1 is on");
    Require(q::ParseIqWmmaMode("2").mode == Mode::kPair2 &&
                !q::ParseIqWmmaMode("2").unsupported,
            "2 is the pair route");
    for (const char* bad : {"", "3", "02", "2 ", " 1", "true", "on", "-1"}) {
      const auto parsed = q::ParseIqWmmaMode(bad);
      Require(parsed.mode == Mode::kPair2 && parsed.unsupported,
              "unknown value must warn and keep the default 2");
    }
    static_assert(q::ParseIqWmmaMode("2").mode == Mode::kPair2);

    // Coverage: value 2, gate == up of a covered type, wide 128-row pair
    // tiles, k a multiple of 256. The model shape is hidden 2560, ff 640.
    constexpr std::uint32_t kIq2S = 22, kIq2Xxs = 16, kIq2Xs = 17,
                            kIq3Xxs = 18, kIq3S = 21, kIq4Xs = 23, kQ4K = 12;
    Require(q::IqPair2Covers(Mode::kPair2, kIq2S, kIq2S, 128, 2560, 640),
            "IQ2_S wide gate/up covered");
    Require(!q::IqPair2Covers(Mode::kOn, kIq2S, kIq2S, 128, 2560, 640),
            "value 1 never takes the pair kernel");
    Require(!q::IqPair2Covers(Mode::kOff, kIq2S, kIq2S, 128, 2560, 640),
            "value 0 never takes the pair kernel");
    Require(!q::IqPair2Covers(Mode::kPair2, kIq2S, kIq2S, 64, 2560, 640),
            "narrow 64-row pair tiles keep route 1");
    Require(!q::IqPair2Covers(Mode::kPair2, kIq2S, kIq3S, 128, 2560, 640),
            "mixed gate/up types keep route 1");
    Require(!q::IqPair2Covers(Mode::kPair2, kIq2S, kIq2S, 128, 2560 + 64, 640),
            "k off the 256 grid keeps route 1");
    Require(!q::IqPair2Covers(Mode::kPair2, kIq2S, kIq2S, 128, 0, 640) &&
                !q::IqPair2Covers(Mode::kPair2, kIq2S, kIq2S, 128, 2560, 0),
            "empty shapes keep route 1");
    for (const std::uint32_t t : {kIq3Xxs, kIq3S, kIq2Xs, kIq2Xxs})
      Require(q::IqPair2Covers(Mode::kPair2, t, t, 128, 2560, 640) &&
                  !q::IqPair2Covers(Mode::kPair2, t, t, 64, 2560, 640),
              "IQ3 (stage 2) and IQ2_XS/XXS (stage 3) wide gate/up covered, narrow tiles not");
    // IQ4_XS stays route 1 (not in the native files' gate/up); IQ1_M has no WMMA path.
    for (const std::uint32_t t : {kIq4Xs, kQ4K})
      Require(!q::IqPair2Covers(Mode::kPair2, t, t, 128, 2560, 640),
              "type not covered by this stage");
    std::cout << "IQ ROUTE POLICY PASS\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "IQ ROUTE POLICY FAIL: " << e.what() << '\n';
    return 1;
  }
}
