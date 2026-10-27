#include <filesystem>
#include <iostream>
#include <string_view>
#include <vector>

#include "src/core/prepare/fingerprint_table.hpp"
#include "src/core/prepare/metadata_json.hpp"
#include "src/core/prepare/model_notices.hpp"
#include "src/core/prepare/release_evidence.hpp"

namespace {
using namespace bubo::prepare;
// Evidence trees for one model set, in catalogue order of source repositories.
struct SetGate {
  ModelSet set;
  std::vector<std::string_view> evidence_files;
};
const SetGate kSets[] = {
    {ModelSet::Huihui, {"huihui.json", "unsloth.json", "bubo.json"}},
    {ModelSet::Original, {"unsloth.json", "bubo-0a3c343b.json"}},
    {ModelSet::LowMemory, {"unsloth.json", "bubo.json"}},
};
void CheckNotices(ModelSet set) {
  // E4: the approved LICENSE, NOTICE and PROVENANCE.md bundle of each set must
  // be embedded.
  const auto& notices = ModelNotices(set);
  constexpr std::string_view names[] = {"LICENSE", "NOTICE", "PROVENANCE.md"};
  if (notices.size() != 3)
    throw std::invalid_argument("E4 model notice bundle is OPEN");
  for (std::size_t i = 0; i < 3; ++i)
    if (notices[i].name != names[i] || notices[i].text.empty() ||
        notices[i].sha256.size() != 64)
      throw std::invalid_argument("E4 model notice bundle is malformed");
}
}  // namespace

int main(int argc, char* argv[]) {
  if (argc == 2 && std::string_view(argv[1]) == "--development") {
    for (const auto& gate : kSets) {
      const auto error = ValidateFields(ModelCatalogue(gate.set), false);
      if (!error.empty()) {
        std::cout << "FAIL: " << ModelSetName(gate.set) << ": " << error
                  << "\n";
        return 1;
      }
    }
    std::cout
        << "PASS: development catalogue (huihui, original and lowmemory sets); "
           "production pins remain OPEN\n";
    return 0;
  }
  if (argc != 3 || std::string_view(argv[1]) != "--release") {
    std::cerr << "Usage: prepare_gate --development | --release EVIDENCE_DIR\n";
    return 2;
  }
  try {
    for (const auto& gate : kSets) {
      const std::string label(ModelSetName(gate.set));
      try {
        const auto& entries = ModelCatalogue(gate.set);
        const auto fields = ValidateFields(entries, true);
        if (!fields.empty())
          throw std::invalid_argument(fields);
        CheckNotices(gate.set);
        std::vector<SourceEvidence> evidence;
        for (auto name : gate.evidence_files) {
          const auto path =
              std::filesystem::u8path(argv[2]) / std::string(name);
          const auto parsed = LoadSourceEvidence(metadata::Read(path));
          if (!parsed)
            throw std::invalid_argument(std::string(name) + ": " +
                                        parsed.error);
          evidence.push_back(parsed.evidence);
        }
        const bool lowmemory = gate.set == ModelSet::LowMemory;
        if (lowmemory) {
          const auto proof = metadata::Read(std::filesystem::u8path(argv[2]) /
                                            "lowmemory-local.json");
          const auto error = ValidateLowMemoryLocalEvidence(proof);
          if (!error.empty())
            throw std::invalid_argument(error);
          for (const auto& entry : entries)
            if (entry.kind == Kind::MainShard) {
              const auto fingerprint = ProductionFingerprintTable(entry);
              if (!fingerprint)
                throw std::invalid_argument(fingerprint.error);
            }
        }
        // LowMemory main shards have local evidence. Its unchanged auxiliary
        // files still require the same immutable HF evidence as Huihui.
        const auto checked = std::span<const Entry>(entries);
        const auto error = ValidateReleaseEvidence(
            lowmemory ? checked.subspan(4) : checked, evidence);
        if (!error.empty())
          throw std::invalid_argument(error);
      } catch (const std::exception& error) {
        throw std::invalid_argument(label + " set: " + error.what());
      }
      std::cout
          << "PASS: " << ModelSetName(gate.set)
          << (gate.set == ModelSet::LowMemory
                  ? " set: four local composed pins/fingerprints and four "
                    "immutable auxiliary pins; hosting is unpublished\n"
                  : " set: all eight pins match immutable source tree "
                    "evidence\n");
    }
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: prepare release gate: " << error.what() << '\n';
    return 1;
  }
}
