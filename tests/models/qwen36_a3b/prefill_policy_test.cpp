#include "src/models/qwen36_a3b/prefill_policy.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace qfn = gufo::models::qwen38_flash_next;
namespace a3b = gufo::models::qwen36_a3b;
namespace {
void Check(bool ok) {
  if (!ok)
    throw std::runtime_error("prefill policy contract");
}
template<class F>
void Reject(F f) {
  bool rejected = false;
  try {
    f();
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  Check(rejected);
}
auto Parse(const char* k, const char* mode = nullptr,
           const char* mass = nullptr, const char* fold = nullptr,
           const char* audit = nullptr) {
  return qfn::ParsePrefillExperts(k, mode, mass, fold, audit, 8, 40);
}
}  // namespace
int main() {
  try {
    const auto off = Parse(nullptr);
    Check(!off.active && off.keep == 8 && off.IdentityLine().empty());
    Check(qfn::PrefillExpertLogLine(off, {}, 0, 7, 12, true).empty());
    const std::vector<std::uint8_t> baseline{1, 2, 0, 255};
    Check(a3b::PrefillCacheIdentity(baseline, off) == baseline);
    Check(!Parse(nullptr, nullptr, "nan", "absent", "bad").active);
    Check(Parse(nullptr, "ref10").keep == 8);
    for (auto k : {"0", "9", "-1", "8x", "4294967296", ""})
      Reject([&] { Parse(k); });
    Reject([] { Parse("4", "unknown"); });
    Reject([] { Parse("4", nullptr, nullptr, nullptr, "unknown"); });
    const auto path = std::filesystem::current_path() / "a3b-fold-test.txt";
    const auto name = path.string();
    for (int count : {39, 41, 48}) {
      {
        std::ofstream out(path);
        for (int i = 0; i < count; ++i)
          out << "1\n";
      }
      Reject([&] { Parse("4", "fold", nullptr, name.c_str()); });
    }
    {
      std::ofstream out(path);
      for (int i = 0; i < 40; ++i)
        out << "1\n";
    }
    const auto fold = Parse("4", "fold", nullptr, name.c_str());
    Check(fold.fold.size() == 40 && fold.fold[39] == 1);
    Check(fold.fold_sha256 ==
          "a834ffb02981aa1f080d8686bd1f1f8f571e895ef8a6deef865cf90023a76e19");
    for (const auto& policy :
         {Parse("4"), Parse("3", "ref10"), fold, Parse("8", "mass", "0.75")}) {
      qfn::PrefillDiagTotals totals;
      totals.routed_experts = {7 * 40 * policy.keep, 7 * 40 * 8, 0, 0};
      auto expected = baseline;
      const auto identity = policy.IdentityLine();
      expected.insert(expected.end(), identity.begin(), identity.end());
      Check(a3b::PrefillCacheIdentity(baseline, policy) == expected);
      const auto log =
          qfn::PrefillExpertLogLine(policy, totals, 32, 7, 12, true);
      Check(log.starts_with(identity.substr(0, identity.size() - 1) +
                            " start=32 tokens=7 "));
      Check(log.find("expert_total_slots=2240") != std::string::npos);
      auto full = policy;
      full.audit = qfn::PrefillExpertsAudit::kFull;
      Check(full.IdentityLine() == policy.IdentityLine());
      totals.routed_experts.executed_rows = totals.routed_experts.active_slots;
      Check(qfn::PrefillExpertLogLine(full, totals, 32, 7, 12, true)
                .ends_with("expert_invalid_rows=0 expert_counter_ok=1"));
    }
    Check(qfn::ParsePrefillExperts(nullptr, "ref10", nullptr, nullptr).keep ==
          10);
    {
      std::ofstream out(path);
      for (int i = 0; i < 40; ++i)
        out << (i == 39 ? "nan\n" : "1\n");
    }
    Reject([&] { Parse("4", "fold", nullptr, name.c_str()); });
    std::filesystem::remove(path);
    std::cout
        << "PASS: used=8/layers=40 settings, identity and developer logs\n";
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
}
