#include "src/models/qwen38_flash_next/prefill_telemetry.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "src/models/qwen38_flash_next/continuation_identity.hpp"
#include "src/models/qwen38_flash_next/prefill_diag.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
namespace {
void Require(bool ok, const char* message) {
  if (!ok)
    throw std::runtime_error(message);
}
template<typename F>
void Reject(F f) {
  bool rejected = false;
  try {
    f();
  } catch (const std::exception&) {
    rejected = true;
  }
  Require(rejected, "invalid telemetry input accepted");
}
void Near(float actual, float expected) {
  Require(std::abs(actual - expected) < 1e-6f, "float mismatch");
}
}  // namespace

int main() {
  try {
    // A stray layers override must not activate telemetry or change log/cache
    // strings. The existing c4_contracts test freezes the complete identities.
    const auto off = qfn::ParsePrefillTelemetry(nullptr, "bad,layers");
    Require(!off.enabled, "telemetry activated while unset");
    Require(!qfn::MakePrefillTelemetryPass(off, 7, 100, 2, "reference:10"),
            "off writer allocated");
    const auto experts =
        qfn::ParsePrefillExperts(nullptr, nullptr, nullptr, nullptr);
    Require(experts.IdentityLine().empty(), "default identity changed");
    Require(qfn::PrefillExpertLogLine(experts, {}, 100, 2, 1, true).empty(),
            "default log changed");
    Reject([] { qfn::ParsePrefillTelemetry("", nullptr); });
    for (const char* text : {"", "3,", "-1", "48", "3,3", "3,x", "3,,15"})
      Reject([&] { qfn::ParsePrefillTelemetry("unused", text); });
    const auto defaults = qfn::ParsePrefillTelemetry("unused", nullptr);
    for (unsigned l = 0; l < 48; ++l)
      Require(defaults.layers[l] ==
                  (l == 3 || l == 15 || l == 27 || l == 39 || l == 47),
              "default layers changed");

    // Hand case: nonzero counts 1,16,17,65. At 16 rows: 1+1+2+5=9;
    // at 64 rows: 1+1+1+2=5. Paired gate/up + down = 14 list entries.
    const std::array<std::uint32_t, 6> counts{0, 1, 16, 17, 65, 0};
    const auto population = qfn::TelemetryPopulation(counts);
    Require(
        population.touched == 4 && population.p50 == 16 && population.p90 == 65,
        "population quantile mismatch");
    Require(qfn::PackedRowTiles(counts, 16) == 9 &&
                qfn::PackedRowTiles(counts, 64) == 5,
            "padded tile count mismatch");
    Reject([&] { qfn::PackedRowTiles(counts, 0); });
    const std::array<float, 10> weights{.4f,  .2f,   .1f,   .1f,   .05f,
                                        .05f, .025f, .025f, .025f, .025f};
    // Two inject partials per stream; all sums zero => four effective gates
    // exactly 1, L2=2. These are logits/partials, not already sigmoid gates.
    const std::array<float, 8> inject{1, -1, 2, -2, 3, -3, 4, -4};
    const auto record = qfn::MakeTelemetryRecord(101, 3, weights, inject, 2);
    Near(record.values[0], .4f);
    Near(record.values[1], .8f);
    Near(record.values[2], 2);
    for (unsigned i = 3; i < 7; ++i)
      Near(record.values[i], 1);

    const auto root =
        std::filesystem::temp_directory_path() / "gufo-prefill-telemetry-cpu";
    std::filesystem::create_directories(root);
    auto enabled = qfn::ParsePrefillTelemetry(root.string().c_str(), "3");
    auto pass =
        qfn::MakePrefillTelemetryPass(enabled, 7, 100, 2, "reference:10");
    auto incomplete =
        qfn::MakePrefillTelemetryPass(enabled, 6, 100, 2, "reference:10");
    incomplete->RecordLayer(3, 100, 2, counts, "f16-wmma-paired", 16, 64, 1, 9,
                            5);
    Reject([&] { incomplete->Finish(); });
    for (unsigned layer = 0; layer < 48; ++layer)
      pass->RecordLayer(layer, 100, 2, counts, "f16-wmma-paired", 16, 64, 1, 9,
                        5);
    pass->Record(record);
    auto first = record;
    first.token = 100;
    pass->Record(first);
    const auto index = pass->Finish();
    const auto layer_metadata = gufo::json::parse([&] {
      std::ifstream in(index.parent_path() / "layers.jsonl");
      std::string line;
      std::getline(in, line);
      return line;
    }());
    Require(layer_metadata.find("row_tiles_launched") &&
                layer_metadata.find("row_tiles_launched")->as_size() == 14,
            "total row tiles must include gate/up and down dispatches");
    const auto metadata = gufo::json::parse([&] {
      std::ifstream in(index);
      return std::string(std::istreambuf_iterator<char>(in), {});
    }());
    const auto binary = index.parent_path() / metadata.find("binary")->str();
    const auto parsed = qfn::ReadTelemetryRecords(binary, 7);
    Require(parsed.size() == 2 && parsed[0].token == 101 &&
                parsed[0].layer == 3 && parsed[0].values == record.values,
            "binary round trip mismatch");
    Require(std::filesystem::file_size(binary) == 32 + 2 * 36,
            "unexpected binary layout size");
    Reject([&] { qfn::ReadTelemetryRecords(binary, 8); });
    const auto truncated = index.parent_path() / "truncated.bin";
    std::filesystem::copy_file(binary, truncated);
    std::filesystem::resize_file(truncated, 33);
    Reject([&] { qfn::ReadTelemetryRecords(truncated, 7); });
    std::cout << "fixture_index=" << index.string() << '\n';
    std::cout << "PASS telemetry settings, off strings, populations, gates, "
                 "tiles, binary round trip and truncation\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
