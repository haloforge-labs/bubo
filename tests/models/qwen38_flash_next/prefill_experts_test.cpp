#include "src/models/qwen38_flash_next/prefill_experts.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <vector>

#include "src/models/qwen38_flash_next/continuation_identity.hpp"
#include "src/models/qwen38_flash_next/prefill_diag.hpp"

namespace qfn = gufo::models::qwen38_flash_next;
namespace {
void Require(bool ok, const char* message) {
  if (!ok)
    throw std::runtime_error(message);
}
void Near(float actual, float expected) {
  Require(std::abs(actual - expected) < 1e-6f, "weight mismatch");
}
// Independent CPU double: rank probabilities with the GPU's low-id tie rule,
// then apply the four policies. Inputs are already softmax probabilities.
struct Routing {
  std::vector<int> ids;
  std::vector<float> weights;
};
Routing Route(std::vector<float> p, unsigned k, unsigned keep,
              qfn::PrefillExpertsMode mode, float mass = 1, float fold = 1) {
  Routing r;
  r.ids.resize(p.size());
  std::iota(r.ids.begin(), r.ids.end(), 0);
  std::sort(r.ids.begin(), r.ids.end(), [&](int a, int b) {
    return p[a] > p[b] || (p[a] == p[b] && a < b);
  });
  r.ids.resize(k);
  float reference = 0;
  for (auto id : r.ids)
    reference += p[id];
  reference = std::max(reference, 6.103515625e-5f);
  if (mode == qfn::PrefillExpertsMode::kMass) {
    const float threshold =
        mass == 1 ? mass - 4 * std::numeric_limits<float>::epsilon() : mass;
    float cumulative = 0;
    unsigned count = 0;
    do {
      cumulative += p[r.ids[count++]] / reference;
    } while (count < keep && cumulative < threshold);
    keep = count;
  }
  float denominator = reference;
  if (mode == qfn::PrefillExpertsMode::kRenorm ||
      mode == qfn::PrefillExpertsMode::kMass) {
    denominator = 0;
    for (unsigned s = 0; s < keep; ++s)
      denominator += p[r.ids[s]];
    denominator = std::max(denominator, 6.103515625e-5f);
  }
  for (unsigned s = 0; s < k; ++s) {
    r.weights.push_back(s < keep ? p[r.ids[s]] / denominator : 0);
    if (s < keep && mode == qfn::PrefillExpertsMode::kFold)
      r.weights.back() *= fold;
    if (s >= keep)
      r.ids[s] = -1;
  }
  return r;
}
void RouterCases() {
  using M = qfn::PrefillExpertsMode;
  const std::vector<float> p{0.125f, 0.5f, 0.25f, 0.125f};
  const auto full = Route(p, 4, 4, M::kRenorm);
  Require(full.ids == std::vector<int>({1, 2, 0, 3}), "ranking/ties");
  for (unsigned i = 0; i < 4; ++i)
    Require(std::bit_cast<unsigned>(full.weights[i]) ==
                std::bit_cast<unsigned>(p[full.ids[i]]),
            "keep=k changed old weights");
  auto r = Route(p, 4, 2, M::kRenorm);
  Near(r.weights[0], 2.0f / 3);
  Near(r.weights[1], 1.0f / 3);
  Require(r.ids[2] == -1 && r.weights[2] == 0, "mask");
  r = Route(p, 4, 1, M::kRenorm);
  Near(r.weights[0], 1);
  r = Route(p, 4, 2, M::kRef10);
  Near(r.weights[0], 0.5f);
  Near(r.weights[1], 0.25f);
  r = Route(p, 4, 2, M::kFold, 1, 2);
  Near(r.weights[0], 1);
  Near(r.weights[1], 0.5f);
  r = Route(p, 4, 4, M::kMass, 0.75f);
  Near(r.weights[0], 2.0f / 3);
  Require(r.ids[2] == -1, "mass boundary");
  r = Route(p, 4, 4, M::kMass, 1);
  Require(r.ids == full.ids && r.weights == full.weights, "mass=1");
  r = Route({1, 0, 0, 0}, 4, 4, M::kMass, 1);
  Require(r.ids == std::vector<int>({0, -1, -1, -1}),
          "mass=1 saturated prefix must keep one");
  Near(r.weights[0], 1);
  r = Route(p, 4, 1, M::kMass, 0.99f);
  Near(r.weights[0], 1);
  r = Route(std::vector<float>(10, 0.1f), 10, 10, M::kMass, 0.25f);
  Require(r.ids[2] == 2 && r.ids[3] == -1, "equal mass/ties");
  Near(r.weights[0], 1.0f / 3);
  // A real top-10 reference in a larger expert population.
  r = Route(std::vector<float>(512, 1.0f / 512), 10, 2, M::kRef10);
  Near(r.weights[0], 0.1f);
  Require(r.ids[0] == 0 && r.ids[1] == 1 && r.ids[2] == -1, "top10 ties");
}
template<class F>
void Reject(F f) {
  bool rejected = false;
  try {
    f();
  } catch (const std::exception&) {
    rejected = true;
  }
  Require(rejected, "invalid setting accepted");
}
void ConfigAndIdentity() {
  const auto off = qfn::ParsePrefillExperts(nullptr, nullptr, nullptr, nullptr);
  Require(!off.active && off.IdentityLine().empty(), "default not off");
  const auto renorm = qfn::ParsePrefillExperts("4", nullptr, nullptr, nullptr);
  const auto ref = qfn::ParsePrefillExperts("4", "ref10", nullptr, nullptr);
  const auto mass = qfn::ParsePrefillExperts(nullptr, "mass", "0.75", nullptr);
  Require(renorm.audit == qfn::PrefillExpertsAudit::kLean,
          "active audit default must be lean");
  const auto full_audit =
      qfn::ParsePrefillExperts("4", nullptr, nullptr, nullptr, "full");
  Require(full_audit.audit == qfn::PrefillExpertsAudit::kFull &&
              full_audit.IdentityLine() == renorm.IdentityLine(),
          "audit must not change cache identity");
  Require(qfn::ParsePrefillExperts("4", nullptr, nullptr, nullptr, "lean")
                  .IdentityLine() == renorm.IdentityLine(),
          "explicit lean identity");
  Reject([] {
    qfn::ParsePrefillExperts("4", nullptr, nullptr, nullptr, "unknown");
  });
  Reject([] { qfn::ParsePrefillExperts("4", nullptr, nullptr, nullptr, ""); });
  Require(
      !qfn::ParsePrefillExperts(nullptr, nullptr, nullptr, nullptr, "unknown")
           .active,
      "audit auxiliary must stay inactive when off");
  Require(renorm.IdentityLine() == "prefill_experts=renorm:4\n",
          "renorm identity");
  Require(ref.IdentityLine() != renorm.IdentityLine(), "variant identity");
  Require(mass.keep == 10 && mass.mass == 0.75f, "mass config");
  for (auto value : {"0", "11", "-1", "2x", "", "4294967296"})
    Reject([&] { qfn::ParsePrefillExperts(value, nullptr, nullptr, nullptr); });
  Reject([] { qfn::ParsePrefillExperts("4", "other", nullptr, nullptr); });
  for (auto value : {"0", "1.1", "nan", "inf", "0.5x", ""})
    Reject([&] { qfn::ParsePrefillExperts("4", "mass", value, nullptr); });
  Reject([] { qfn::ParsePrefillExperts("4", "fold", nullptr, nullptr); });
  Require(!qfn::ParsePrefillExperts(nullptr, nullptr, "0.5", "absent").active,
          "auxiliary settings activated the knob");
  const auto path = std::filesystem::current_path() / "qfn-kcut-fold-test.txt";
  const auto name = path.string();
  Reject([&] { qfn::ParsePrefillExperts("2", "fold", nullptr, name.c_str()); });
  for (auto bad : {"1\n", "nan\n"}) {
    std::ofstream(path) << bad;
    Reject(
        [&] { qfn::ParsePrefillExperts("2", "fold", nullptr, name.c_str()); });
  }
  {
    std::ofstream out(path);
    for (int i = 0; i < 48; ++i)
      out << "1.25\n";
  }
  const auto fold =
      qfn::ParsePrefillExperts("2", "fold", nullptr, name.c_str());
  Near(fold.fold[47], 1.25f);
  {
    std::ofstream out(path);
    for (int i = 0; i < 48; ++i)
      out << "1.5\n";
  }
  const auto changed =
      qfn::ParsePrefillExperts("2", "fold", nullptr, name.c_str());
  Require(fold.IdentityLine() != changed.IdentityLine(),
          "fold contents not keyed");
  qfn::PrefillDiagTotals lean;
  lean.routed_experts = {4, 10, 0, 0};
  const auto mass_log = qfn::PrefillExpertLogLine(mass, lean, 0, 1, 12, true);
  Require(mass_log.find("prefill_experts=mass:10:tau_bits=3f400000 ") == 0 &&
              mass_log.find("audit=lean") != std::string::npos,
          "developer line missing canonical tau/audit");
  const auto fold_log = qfn::PrefillExpertLogLine(fold, lean, 0, 1, 12, true);
  Require(fold_log.starts_with(
              fold.IdentityLine().substr(0, fold.IdentityLine().size() - 1)) &&
              fold_log.find(":scalars_sha256=" + fold.fold_sha256) !=
                  std::string::npos,
          "developer line missing canonical fold contents");
  std::filesystem::remove(path);
  // Frozen c4 canonical text, with controlled artifact/template arguments.
  const std::string baseline =
      "schema=gufo-text-continuation-v2\nmodel_kind=qwen38-flash-next\n"
      "artifact_id=gguf-sha256-v1:artifact\ntokenizer=embedded-in-artifact\n"
      "chat_template=qwen38-reasoning-compiled-v3\n"
      "chat_template_reference_sha256=template\n"
      "state_abi=qwen38-flash-next-rocm-session-v1\n"
      "payload_layout=qfn-rocm-session-snapshot-v21\ncontext_tokens=32768\n"
      "position_policy=absolute-v1\nadapters=none\n" +
      qfn::KvCacheCompatibilityIdentity(qfn::KvCacheType::kF16);
  const auto identity = [&](const auto& settings, bool mtp = false,
                            qfn::KvCacheType type = qfn::KvCacheType::kF16) {
    return qfn::FlashNextCompatibilityIdentity(
        "artifact", "draft", mtp, 32768, 7, 4, 21, "template", settings, type);
  };
  Require(identity(off) == baseline, "default c4 identity changed");
  Require(identity(off, true) ==
              baseline +
                  "draft_backend=qfn-mtp-v1\ndraft_artifact_id=gguf-sha256-v1:"
                  "draft\n"
                  "draft_max_tokens=7\ndraft_cost_concurrency=4\n",
          "default MTP identity changed");
  Require(identity(renorm) == baseline + renorm.IdentityLine(),
          "active identity");
  Require(identity(renorm) != identity(off), "active matches off");
  Require(identity(renorm, true) == identity(off, true) + renorm.IdentityLine(),
          "active MTP identity");
  for (const auto type :
       {qfn::KvCacheType::kQ8_0, qfn::KvCacheType::kQ8_0Rotated}) {
    const auto packed = identity(off, false, type);
    Require(packed != identity(off), "KV dtype shares the F16 cache key");
    Require(packed.find(std::string("ordinary_kv_dtype=") +
                        qfn::KvCacheTypeName(type) + "\n") != std::string::npos,
            "KV dtype missing from the canonical key");
    Require(identity(renorm, false, type) == packed + renorm.IdentityLine(),
            "KV dtype displaced the fast-mode row policy");
    Require(identity(renorm, false, type) != identity(off, false, type),
            "fast-mode row policy shares the full-row cache key");
  }
  qfn::PrefillDiagTotals totals;
  Require(
      qfn::PrefillDiagFields(totals) ==
          "table_read_us=0.000 table_reads=0 table_bytes=0 "
          "fetch_elapsed_us=0.000 "
          "fetch_hidden_us=0.000 fetch_exposed_us=0.000 fetch_wait_us=0.000 "
          "experts_distinct_min=0 experts_distinct_mean=0.000 "
          "experts_distinct_max=0 expert_bytes=0",
      "default developer log fields changed");
  totals.routed_experts = {20, 100, 20, 0};
  totals.expert_audit = qfn::PrefillExpertsAudit::kFull;
  Require(qfn::PrefillExpertFields(totals) ==
              " audit=full expert_active_slots=20 expert_total_slots=100 "
              "expert_rows_executed=20 expert_invalid_rows=0",
          "counter field contract");
}
void TailSettings() {
  const auto parse = [](const char* tail, const char* count = "2",
                        const char* audit = nullptr) {
    return qfn::ParsePrefillExperts(count, nullptr, nullptr, nullptr, audit, 10,
                                    48, tail);
  };
  Require(parse(nullptr).tail == 0, "unset tail");
  Require(parse("0").IdentityLine() == parse(nullptr).IdentityLine(),
          "zero changed identity");
  Require(parse("512").IdentityLine() == "prefill_experts=renorm:2:tail=512\n",
          "tail identity");
  Require(parse("65536").tail == 65536, "maximum tail");
  Reject([] {
    (void)qfn::ParsePrefillExperts("2", nullptr, nullptr, nullptr, nullptr, 8,
                                   40, "512");
  });
  Require(parse("512", "2", "full").IdentityLine() ==
              parse("512", "2", "lean").IdentityLine(),
          "audit leaked into tail identity");
  Require(!parse("junk", nullptr).active &&
              parse("512", nullptr).IdentityLine().empty() &&
              parse("512", nullptr).tail == 0,
          "tail auxiliary activated policy");
  for (const auto* bad : {"", "-1", "+1", " 1", "1 ", "1x", "65537",
                          "4294967296", "9999999999999999999999"})
    Reject([&] { (void)parse(bad); });
  const auto mass = qfn::ParsePrefillExperts("2", "mass", "0.75", nullptr,
                                             nullptr, 10, 48, "2048");
  Require(mass.IdentityLine() ==
              "prefill_experts=mass:2:tau_bits=3f400000:tail=2048\n",
          "tail must stay decimal after tau bits");
  const auto identity = [&](const auto& s) {
    return qfn::FlashNextCompatibilityIdentity("artifact", "draft", false,
                                               32768, 7, 4, 21, "template", s);
  };
  Require(identity(parse("0")) == identity(parse(nullptr)), "cache zero");
  Require(identity(parse("256")) != identity(parse("512")), "cache tail");
  const auto line = qfn::PrefillExpertLogLine(parse("512"), {}, 0, 8, 1, true);
  Require(line.starts_with("prefill_experts=renorm:2:tail=512 "), "diag tail");
}
void CountedWork() {
  // Three full-width invocations and a two-token final tail; the variable
  // mass counts are deliberately unrelated to token_count * cap.
  const std::vector<std::uint32_t> counts{0, 7, 0, 3, 2};
  Require(qfn::CountRoutedExpertSlots(counts) == 12, "live count sum");
  qfn::PrefillDiagTotals lean;
  lean.routed_experts = {qfn::CountRoutedExpertSlots(counts), 40, 0, 0};
  Require(qfn::PrefillExpertFields(lean) ==
              " audit=lean expert_active_slots=12 expert_total_slots=40 "
              "expert_rows_counted=12",
          "lean must log counts without claiming observed writes");
  Require(qfn::MoeFlopsForRows(4, 16, 8) == 0.4 * qfn::MoeFlops(1, 10, 16, 8),
          "k=4 logical flops must be 0.4 of full k=10");
  const std::vector<std::uint32_t> tail_counts{1, 0, 2};
  Require(qfn::MoeFlopsForRows(qfn::CountRoutedExpertSlots(counts) +
                                   qfn::CountRoutedExpertSlots(tail_counts),
                               16, 8) == 15 * 6 * 16 * 8,
          "mass and final-tail live flops");
  const auto line = qfn::PrefillExpertLogLine(
      qfn::ParsePrefillExperts("4", nullptr, nullptr, nullptr), lean, 0, 4, 12,
      true);
  Require(line.find("expert_counter_ok") == std::string::npos &&
              line.find("expert_invalid_rows") == std::string::npos &&
              line.find("expert_rows_executed") == std::string::npos,
          "lean line fabricates write evidence");
}
std::string KFileText(const std::string& value, unsigned layers = 48) {
  std::string text;
  for (unsigned layer = 0; layer < layers; ++layer)
    text += value + '\n';
  return text;
}
void KFileSettings() {
  const auto root = std::filesystem::current_path();
  const auto path = root / "qfn-kfile-cpu-test.txt";
  const auto other_path = root / "qfn-kfile-copy-cpu-test.txt";
  const auto fold_path = root / "qfn-kfile-fold-cpu-test.txt";
  const auto name = path.string(), other = other_path.string();
  const auto fold_name = fold_path.string();
  const auto write = [](const auto& target, const std::string& text) {
    std::ofstream out(target, std::ios::binary | std::ios::trunc);
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
    Require(static_cast<bool>(out), "cannot write kfile CPU fixture");
  };
  const auto parse = [&](const char* count = nullptr,
                         const char* mode = nullptr, const char* mass = nullptr,
                         const char* fold = nullptr, const char* tail = nullptr,
                         const char* kfile = nullptr,
                         const char* audit = nullptr) {
    return qfn::ParsePrefillExperts(count, mode, mass, fold, audit, 10, 48,
                                    tail, kfile ? kfile : name.c_str());
  };
  // Missing/unreadable and empty paths must fail rather than disable policy.
  std::filesystem::remove(path);
  Reject([&] { (void)parse(); });
  Reject([&] { (void)parse(nullptr, nullptr, nullptr, nullptr, nullptr, ""); });
  write(path, KFileText("3"));
  const auto uniform = parse();
  Require(uniform.active && uniform.keep == 10 &&
              uniform.mode == qfn::PrefillExpertsMode::kRenorm &&
              uniform.layer_keep.size() == 48,
          "kfile alone did not activate renorm policy");
  for (unsigned layer = 0; layer < 48; ++layer)
    Require(uniform.Keep(layer) == 3, "uniform per-layer keep");
  const std::string u3_hash =
      "e7972e8c782abb8ba7ee495b9b9a945ff99398a18baf5013e3a6e6467c6d31e8";
  Require(uniform.kfile_sha256 == u3_hash,
          "kfile hash must use canonical little-endian u32 values");
  Require(uniform.IdentityLine() ==
              "prefill_experts=renorm:10:kfile_sha256=" + u3_hash + '\n',
          "canonical kfile identity");
  Require(parse(nullptr, "ref10").IdentityLine() ==
              "prefill_experts=ref10:10:kfile_sha256=" + u3_hash + '\n',
          "kfile ref10 mode");
  Require(
      parse(nullptr, "ref10", nullptr, nullptr, "512").IdentityLine() ==
          "prefill_experts=ref10:10:kfile_sha256=" + u3_hash + ":tail=512\n",
      "kfile/tail identity");
  Require(parse(nullptr, nullptr, nullptr, nullptr, "0").IdentityLine() ==
              uniform.IdentityLine(),
          "zero tail changed kfile identity");
  Require(parse(nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, "full")
                  .IdentityLine() == uniform.IdentityLine(),
          "audit changed kfile identity");
  write(other_path, KFileText("3"));
  Require(parse(nullptr, nullptr, nullptr, nullptr, nullptr, other.c_str())
                  .IdentityLine() == uniform.IdentityLine(),
          "path leaked into kfile identity");
  write(path, KFileText("03"));
  Require(parse().IdentityLine() == uniform.IdentityLine(),
          "integer spelling leaked into canonical identity");
  for (const auto* value : {"1", "10"}) {
    write(path, KFileText(value));
    Require(parse().Keep(47) == static_cast<unsigned>(std::stoi(value)),
            "valid kfile endpoint rejected");
  }
  // An unterminated last line follows the existing fold reader's contract.
  auto no_final_lf = KFileText("3");
  no_final_lf.pop_back();
  write(path, no_final_lf);
  Require(parse().IdentityLine() == uniform.IdentityLine(),
          "last line without delimiter changed values");
  for (const auto* bad :
       {"", "0", "11", "-1", "+1", "3.0", "3x", " 3", "3 ", "3\t", "3\r",
        "4294967296", "999999999999999999999999999999999"}) {
    write(path, KFileText("3", 17) + bad + '\n' + KFileText("3", 30));
    Reject([&] { (void)parse(); });
  }
  for (const auto& bad :
       {KFileText("3", 47), KFileText("3", 49), KFileText("3") + '\n',
        std::string{}, std::string("\xef\xbb\xbf") + KFileText("3"),
        KFileText("3\r")}) {
    write(path, bad);
    Reject([&] { (void)parse(); });
  }
  // Embedded NUL and a CR without an LF must not be accepted as whitespace.
  write(path, KFileText("3", 47) + std::string("3\0\n", 3));
  Reject([&] { (void)parse(); });
  write(path, KFileText("3", 47) + "3\r");
  Reject([&] { (void)parse(); });
  write(path, KFileText("3"));
  for (const auto* count : {"3", "", "bad"})
    Reject([&] { (void)parse(count); });
  Reject([&] { (void)parse(nullptr, "mass", "0.75"); });
  Reject([&] { (void)parse(nullptr, "mass"); });
  Reject([&] { (void)parse(nullptr, nullptr, "0.75"); });
  Reject([&] {
    (void)qfn::ParsePrefillExperts(nullptr, nullptr, nullptr, nullptr, nullptr,
                                   8, 40, nullptr, name.c_str());
  });
  Reject([&] { (void)parse(nullptr, "fold"); });
  write(fold_path, KFileText("1.25"));
  const auto fold = parse(nullptr, "fold", nullptr, fold_name.c_str());
  Require(fold.fold.size() == 48 && fold.fold[47] == 1.25f &&
              fold.Keep(47) == 3 &&
              fold.IdentityLine().find(":scalars_sha256=" + fold.fold_sha256 +
                                       ":kfile_sha256=" + u3_hash) !=
                  std::string::npos,
          "kfile/fold composition");
  const auto identity = [](const auto& settings) {
    return qfn::FlashNextCompatibilityIdentity(
        "artifact", "draft", false, 32768, 7, 4, 21, "template", settings);
  };
  const auto off = qfn::ParsePrefillExperts(nullptr, nullptr, nullptr, nullptr);
  Require(off.IdentityLine().empty(), "unset default identity changed");
  Require(
      qfn::ParsePrefillExperts("3", "ref10", nullptr, nullptr).IdentityLine() ==
          "prefill_experts=ref10:3\n",
      "scalar default identity changed");
  Require(qfn::ParsePrefillExperts("3", "ref10", nullptr, nullptr, nullptr, 10,
                                   48, "0")
                  .IdentityLine() == "prefill_experts=ref10:3\n",
          "scalar zero-tail identity changed");
  Require(identity(uniform) != identity(off) &&
              identity(uniform).ends_with(uniform.IdentityLine()),
          "kfile did not enter continuation compatibility identity");
  // Same sum/mean, distinct ordered content: layer order belongs to identity.
  write(path, KFileText("6", 6) + KFileText("2", 39) + KFileText("6", 3));
  const auto first_last = parse(nullptr, "ref10", nullptr, nullptr, "512");
  Require(
      first_last.kfile_sha256 ==
          "387b0c9139e1054b30a2c99ebc6da3b86975f8c741fd66ce8c958abe8c911a13",
      "nonuniform canonical hash");
  Require(first_last.Keep(0) == 6 && first_last.Keep(5) == 6 &&
              first_last.Keep(6) == 2 && first_last.Keep(44) == 2 &&
              first_last.Keep(45) == 6 && first_last.Keep(47) == 6,
          "ordered per-layer values");
  write(path, KFileText("2", 39) + KFileText("6", 9));
  Require(parse(nullptr, "ref10", nullptr, nullptr, "512").IdentityLine() !=
              first_last.IdentityLine(),
          "permutation with equal sum collided");
  // One layer may receive the final trimmed rows; use its k, not the mean.
  std::uint64_t expected = 0;
  for (unsigned layer = 0; layer < 48; ++layer)
    expected += qfn::PrefillExpertExpectedSlots(first_last, layer, 8192);
  Require(expected == 8192ULL * 132,
          "per-layer expected active count must be rows times sum(k)");
  expected -= qfn::PrefillExpertExpectedSlots(first_last, 47, 8192);
  expected += qfn::PrefillExpertExpectedSlots(first_last, 47, 128);
  Require(expected == 8192ULL * 126 + 128ULL * 6,
          "trimmed final layer must use its own keep");
  const auto scalar = qfn::ParsePrefillExperts("3", nullptr, nullptr, nullptr);
  Require(qfn::PrefillExpertExpectedSlots(scalar, 47, 8192) == 8192ULL * 3 &&
              qfn::PrefillExpertExpectedSlots(uniform, 47, 8192) ==
                  qfn::PrefillExpertExpectedSlots(scalar, 47, 8192),
          "uniform kfile/scalar expected counts disagree");
  qfn::RoutedExpertCounters observed;
  observed.offered_slots = 10ULL * (47 * 8192 + 128);
  observed.active_slots = expected;
  Require(qfn::PrefillExpertCountsValid(first_last, observed, 10, expected),
          "lean rejected correct per-layer active counts");
  --observed.active_slots;
  Require(!qfn::PrefillExpertCountsValid(first_last, observed, 10, expected),
          "lean accepted missing layer work");
  observed.active_slots = expected;
  auto full = first_last;
  full.audit = qfn::PrefillExpertsAudit::kFull;
  Require(!qfn::PrefillExpertCountsValid(full, observed, 10, expected),
          "full audit accepted unobserved writes");
  observed.executed_rows = expected;
  Require(qfn::PrefillExpertCountsValid(full, observed, 10, expected),
          "full audit rejected correct per-layer writes");
  ++observed.invalid_rows;
  Require(!qfn::PrefillExpertCountsValid(full, observed, 10, expected),
          "full audit accepted invalid row");
  observed = {};
  Require(!qfn::PrefillExpertCountsValid(first_last, observed, 10, 0),
          "zero-work pass accepted");
  Require(qfn::PrefillExpertActiveSlotsValid(first_last, 8192, 8192ULL * 6,
                                             8192ULL * 6) &&
              !qfn::PrefillExpertActiveSlotsValid(first_last, 8192, 8192ULL * 6,
                                                  8192ULL * 2) &&
              !qfn::PrefillExpertActiveSlotsValid(first_last, 8192, 8192ULL * 2,
                                                  8192ULL * 6),
          "compensating wrong per-layer keep must fail each layer");
  const auto mass = qfn::ParsePrefillExperts("4", "mass", "0.75", nullptr);
  Require(qfn::PrefillExpertActiveSlotsValid(mass, 100, 400, 100) &&
              qfn::PrefillExpertActiveSlotsValid(mass, 100, 400, 400) &&
              !qfn::PrefillExpertActiveSlotsValid(mass, 100, 400, 99) &&
              !qfn::PrefillExpertActiveSlotsValid(mass, 100, 400, 401),
          "existing mass active interval changed");
  std::filesystem::remove(path);
  std::filesystem::remove(other_path);
  std::filesystem::remove(fold_path);
}
void KFileCachedReader() {
  struct Environment {
    const char* key;
    std::optional<std::string> previous;
    static void Set(const char* key, const char* value) {
#ifdef _WIN32
      _putenv_s(key, value ? value : "");
#else
      if (value)
        setenv(key, value, 1);
      else
        unsetenv(key);
#endif
    }
    Environment(const char* name, const char* value) : key(name) {
      if (const auto* prior = std::getenv(key))
        previous = prior;
      Set(key, value);
    }
    ~Environment() { Set(key, previous ? previous->c_str() : nullptr); }
  };
  const auto path =
      std::filesystem::current_path() / "qfn-kfile-static-test.txt";
  const auto name = path.string();
  {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << KFileText("3");
  }
  const Environment count("GUFO_PREFILL_EXPERTS", nullptr);
  const Environment mode("GUFO_PREFILL_EXPERTS_MODE", nullptr);
  const Environment mass("GUFO_PREFILL_EXPERTS_MASS", nullptr);
  const Environment fold("GUFO_PREFILL_EXPERTS_FOLD", nullptr);
  const Environment audit("GUFO_PREFILL_EXPERTS_AUDIT", nullptr);
  const Environment tail("GUFO_PREFILL_EXPERTS_TAIL", "512");
  const Environment kfile("GUFO_PREFILL_EXPERTS_KFILE", name.c_str());
  const auto& first = qfn::GetPrefillExperts();
  Require(first.active && first.Keep(47) == 3 && first.tail == 512 &&
              first.mode == qfn::PrefillExpertsMode::kRenorm,
          "environment kfile did not reach cached reader");
  {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << KFileText("6");
  }
  Environment::Set("GUFO_PREFILL_EXPERTS_KFILE", "absent-after-cache");
  const auto& second = qfn::GetPrefillExperts();
  Require(&first == &second && second.Keep(47) == 3,
          "kfile reader did not cache immutable contents");
  std::filesystem::remove(path);
}
}  // namespace
int main() {
  try {
    RouterCases();
    ConfigAndIdentity();
    TailSettings();
    CountedWork();
    KFileSettings();
    KFileCachedReader();
    std::cout
        << "PASS: prefill expert router, strict KFILE, counts and identity\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "FAIL: " << e.what() << '\n';
    return 1;
  }
}
