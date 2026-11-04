#include "src/models/qwen38_flash_next/prefill_stageb.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "src/models/qwen38_flash_next/prefill_diag.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
namespace {
void Require(bool value) {
  if (!value)
    throw std::runtime_error("stage B CPU assertion failed");
}
template<typename F>
void Reject(F f) {
  bool rejected = false;
  try {
    f();
  } catch (const std::exception&) {
    rejected = true;
  }
  Require(rejected);
}
}  // namespace
int main() {
  try {
    const auto off = qfn::ParsePrefillStageB(nullptr, "bad", "0", "bad");
    Require(!off.enabled);
    Require(!qfn::MakePrefillStageBPass(off, 1, 0, 2));
    const auto experts =
        qfn::ParsePrefillExperts(nullptr, nullptr, nullptr, nullptr);
    Require(experts.IdentityLine().empty());
    Require(qfn::PrefillExpertLogLine(experts, {}, 100, 2, 1, true).empty());
    for (const char* dir : {"", "relative", "C:\\bad", "S:relative"})
      Reject([&] { qfn::ParsePrefillStageB(dir, nullptr, nullptr, nullptr); });
    for (const char* text : {"", "48", "-1", "1,1", "1,", "x"})
      Reject([&] {
        qfn::ParsePrefillStageB("S:\\unused", text, nullptr, nullptr);
      });
    for (const char* text : {"", "0", "-1", "1x", "4294967296"}) {
      Reject([&] {
        qfn::ParsePrefillStageB("S:\\unused", nullptr, text, nullptr);
      });
      Reject([&] {
        qfn::ParsePrefillStageB("S:\\unused", nullptr, nullptr, text);
      });
    }
    const auto defaults =
        qfn::ParsePrefillStageB("S:\\unused", nullptr, nullptr, nullptr);
    Require(defaults.stride == 1 && defaults.sample_every == 64);
    for (bool layer : defaults.layers)
      Require(layer);
    qfn::RequireStageBReference(experts);
    Reject([] {
      qfn::RequireStageBReference(
          qfn::ParsePrefillExperts("3", "ref10", nullptr, nullptr));
    });
    // A KFILE keeps the scalar sentinel at 10; every layer must be full.
    auto layered = qfn::ParsePrefillExperts("10", nullptr, nullptr, nullptr);
    layered.layer_keep.assign(48, 10);
    qfn::RequireStageBReference(layered);
    layered.layer_keep[47] = 3;
    Reject([&] { qfn::RequireStageBReference(layered); });
    layered.layer_keep[47] = 10;
    layered.mode = qfn::PrefillExpertsMode::kRef10;
    qfn::RequireStageBReference(layered);
    layered.layer_keep[0] = 2;
    Reject([&] { qfn::RequireStageBReference(layered); });
    layered.layer_keep[0] = 10;
    for (const auto mode :
         {qfn::PrefillExpertsMode::kMass, qfn::PrefillExpertsMode::kFold}) {
      layered.mode = mode;
      Reject([&] { qfn::RequireStageBReference(layered); });
    }
    const auto parent =
        std::filesystem::temp_directory_path() / "gufo-stageb-cpu";
    std::filesystem::create_directories(parent);
    std::filesystem::path root;
    unsigned run = 0;
    do {
      root = parent / ("run-" + std::to_string(++run));
    } while (!std::filesystem::create_directory(root));
    const auto settings =
        qfn::ParsePrefillStageB(root.string().c_str(), "3", "2", "4");
    auto pass = qfn::MakePrefillStageBPass(settings, 7, 100, 9);
    Require(pass->Selected(3) && !pass->Selected(4));
    Reject([&] { pass->Finish({}, {}); });
    std::array<qfn::StageBTotal, 48> totals{};
    totals[3].rows = 5;
    for (unsigned k = 0; k < 5; ++k) {
      totals[3].rf[k] = totals[3].rr[k] = 5;
    }
    const std::array<float, 5> omitted{.625f, .5f, .375f, .25f, 0.f};
    for (unsigned k = 0; k < 5; ++k)
      totals[3].omitted[k] = 5 * omitted[k];
    totals[3].ff = 5;
    totals[3].top1_mass = 1.25;
    totals[3].top4_mass = 3.125;
    totals[3].gate_l2 = 10;
    // Three records at absolute positions 100,104,108, not chunk-relative.
    pass->ObserveLayer(3, 100, 9);
    std::vector<qfn::StageBRow> rows(48 * pass->SampleCapacity());
    for (unsigned i = 0; i < 3; ++i) {
      auto& row = rows[3 * pass->SampleCapacity() + i];
      row.token = 100 + 4 * i;
      row.layer = 3;
      for (unsigned s = 0; s < 4; ++s) {
        row.gates[s] = 1;
        row.omitted[s] = omitted[s];
      }
      row.gate_l2 = 2;
      row.gate_max = 1;
      row.top1_mass = .25f;
      row.top4_mass = .625f;
      row.f_norm = 1;
    }
    pass->Finish(totals, rows);
    Require(std::filesystem::file_size(root / "stageb-totals.jsonl") > 0);
    std::cout << "fixture_directory=" << root.string() << '\n';
    Reject([&] { pass->Finish(totals, rows); });
    const auto bad = root / "not-a-directory";
    {
      std::ofstream out(bad);
      out << "x";
    }
    Reject([&] {
      qfn::MakePrefillStageBPass(
          qfn::ParsePrefillStageB(bad.string().c_str(), nullptr, nullptr,
                                  nullptr),
          8, 0, 1);
    });
    std::cout << "PASS stageb settings, off strings, reference guard, sampling "
                 "and checked writer\n";
  } catch (const std::exception& e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
