#include <filesystem>
#include <iostream>
#include <string_view>
#include <vector>

#include "src/core/prepare/metadata_json.hpp"
#include "src/core/prepare/model_notices.hpp"
#include "src/core/prepare/release_evidence.hpp"

int main(int argc, char* argv[]) {
  using namespace bubo::prepare;
  if (argc == 2 && std::string_view(argv[1]) == "--development") {
    const auto error = ValidateFields(ModelCatalogue(), false);
    std::cout
        << (error.empty()
                ? "PASS: development catalogue; production pins remain OPEN\n"
                : "FAIL: " + error + "\n");
    return error.empty() ? 0 : 1;
  }
  if (argc != 3 || std::string_view(argv[1]) != "--release") {
    std::cerr << "Usage: prepare_gate --development | --release EVIDENCE_DIR\n";
    return 2;
  }
  try {
    const auto fields = ValidateFields(ModelCatalogue(), true);
    if (!fields.empty())
      throw std::invalid_argument(fields);
    // E4: the approved LICENSE, NOTICE and PROVENANCE.md bundle must be embedded.
    const auto& notices = ModelNotices();
    constexpr std::string_view names[] = {"LICENSE", "NOTICE", "PROVENANCE.md"};
    if (notices.size() != 3)
      throw std::invalid_argument("E4 model notice bundle is OPEN");
    for (std::size_t i = 0; i < 3; ++i)
      if (notices[i].name != names[i] || notices[i].text.empty() ||
          notices[i].sha256.size() != 64)
        throw std::invalid_argument("E4 model notice bundle is malformed");
    std::vector<SourceEvidence> evidence;
    for (auto name : {"huihui.json", "unsloth.json", "bubo.json"}) {
      const auto path = std::filesystem::u8path(argv[2]) / name;
      const auto parsed = LoadSourceEvidence(metadata::Read(path));
      if (!parsed)
        throw std::invalid_argument(std::string(name) + ": " + parsed.error);
      evidence.push_back(parsed.evidence);
    }
    const auto error = ValidateReleaseEvidence(ModelCatalogue(), evidence);
    if (!error.empty())
      throw std::invalid_argument(error);
    std::cout << "PASS: all eight pins match immutable source tree evidence\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "FAIL: prepare release gate: " << error.what() << '\n';
    return 1;
  }
}
