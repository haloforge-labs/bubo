// One CPU comparison of raw read-ahead rows and policy decisions.
#include "src/models/qwen38_flash_next/decode_cycle.hpp"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <vector>

#include "src/core/platform/fs.hpp"
#include "src/core/quant/ggml_dequant.hpp"
#include "src/models/qwen/tokenizer.hpp"
#include "src/models/qwen38_flash_next/mtp_policy.hpp"
#include "src/models/qwen38_flash_next/ngram.hpp"
#include "src/models/qwen38_flash_next/ple_decode.hpp"
#include "tests/models/qwen38_flash_next/ple_test_data.hpp"
#include "tests/support/temporary_directory.hpp"

namespace q = gufo::models::qwen38_flash_next;
using Type = gufo::core::GgmlType;

namespace {
void Require(bool ok, const char* message) {
  if (!ok)
    throw std::runtime_error(message);
}

std::unique_ptr<q::NgramTable> Open(const std::filesystem::path& path,
                                    std::uint64_t offset, std::uint64_t rows,
                                    Type type) {
  const auto file = gufo::platform::OpenReadFile(path);
  std::string error;
  auto table = q::NgramTable::Open(file, offset, rows, 160, type, &error);
  gufo::platform::CloseFile(file);
  Require(table != nullptr, error.c_str());
  return table;
}

void Compare(q::NgramTable& table, std::span<const std::uint32_t> ids) {
  auto read = table.ReadAhead(ids);
  Require(read != nullptr, "raw submission failed");
  std::vector<float> old(ids.size() * table.RowDim());
  // The ordinary gather overlaps independent raw jobs on the same pool.
  Require(table.Read(ids, old), "ordinary read failed");
  Require(read->Wait(), "raw read failed");
  const auto raw = read->RawRows();
  std::vector<float> decoded(old.size());
  for (std::size_t row = 0; row < ids.size(); ++row)
    for (std::uint32_t col = 0; col < table.RowDim(); ++col)
      decoded[row * table.RowDim() + col] =
          q::PleRowElement(raw.data() + row * table.RowBytes(), col,
                           table.Type() == Type::kIQ4_NL);
  Require(
      std::memcmp(old.data(), decoded.data(), old.size() * sizeof(float)) == 0,
      "raw decode differs from the existing CPU row path");
}

void CompareChain(q::NgramTable& table, const q::Config& config,
                  std::span<const std::int32_t> tokens) {
  q::NgramHistory before;
  before.prev = {17, 29};
  q::PleReadAhead ahead;
  ahead.Begin(before);
  auto owner = std::make_shared<q::NgramCounters>();
  for (const auto token : tokens)
    Require(ahead.Append(table, config, token, owner),
            "token read-ahead failed");
  Require(before.prev[0] == 17 && before.prev[1] == 29,
          "read-ahead advanced live history");
  Require(ahead.Matches(before, tokens), "identical chain did not match");
  auto wrong = before;
  wrong.prev[0] = 42;
  Require(!ahead.Matches(wrong, tokens), "different history claimed rows");
  std::vector<std::int32_t> changed(tokens.begin(), tokens.end());
  ++changed.back();
  Require(!ahead.Matches(before, changed), "different token claimed rows");
  auto history = before;
  std::vector<std::uint32_t> ids(tokens.size() * config.ple_heads);
  q::HashNgramRows(config, history, tokens, ids);
  std::vector<float> old(ids.size() * table.RowDim());
  Require(table.Read(ids, old), "chain ordinary read failed");
  const auto bytes = ids.size() * table.RowBytes();
  std::vector<std::uint8_t> raw(bytes + 16, 0xA5);
  Require(ahead.CopyRows(tokens.size(), std::span(raw).first(bytes)),
          "chain join failed");
  for (std::size_t i = bytes; i < raw.size(); ++i)
    Require(raw[i] == 0xA5, "raw gather overran its output");
  std::vector<float> decoded(old.size());
  for (std::size_t row = 0; row < ids.size(); ++row)
    for (std::uint32_t col = 0; col < table.RowDim(); ++col)
      decoded[row * table.RowDim() + col] =
          q::PleRowElement(raw.data() + row * table.RowBytes(), col,
                           table.Type() == Type::kIQ4_NL);
  Require(
      std::memcmp(old.data(), decoded.data(), old.size() * sizeof(float)) == 0,
      "incremental chain differs from the existing CPU row path");
  Require(ahead.Totals().issued == ids.size() &&
              ahead.Totals().used == ids.size() && ahead.Totals().wasted == 0,
          "read-ahead counters do not describe the consumed rows");

  // A survival cut may claim a prefix; the unused suffix is wasted once.
  ahead.Begin(before);
  for (const auto token : tokens)
    Require(ahead.Append(table, config, token, owner),
            "prefix submission failed");
  Require(ahead.Matches(before, tokens.first(1)),
          "prefix could not claim rows");
  Require(ahead.CopyRows(1, raw), "prefix join failed");
  Require(ahead.Totals().wasted == (tokens.size() - 1) * config.ple_heads,
          "unused tail was not counted");
  const auto used = ahead.Totals().used;
  const auto wasted = ahead.Totals().wasted;
  ahead.Begin(before);
  Require(ahead.Append(table, config, tokens.front(), owner),
          "cancelled chain submission failed");
  ahead.Discard();
  ahead.Discard();
  Require(!ahead.Matches(before, tokens.first(1)) &&
              ahead.Totals().used == used &&
              ahead.Totals().wasted == wasted + config.ple_heads,
          "abandoned rows remained claimable or were counted twice");
}

q::Config FixtureConfig() {
  q::Config c;
  c.ple_layer = 1;
  c.ple_ngram_size = 3;
  c.ple_heads_per_ngram = 8;
  c.ple_heads = 16;
  c.ple_head_dim = 160;
  c.ple_eos_token = 248044;
  c.ple_multipliers = {23703573157769ULL, 20109073645365ULL, 8052911324071ULL};
  c.ple_head_vocab.fill(251);
  for (std::uint32_t h = 0; h < c.ple_heads; ++h)
    c.ple_head_offsets[h] = h * 251;
  return c;
}

void CpuRows() {
  TemporaryDirectory temporary("decode-cycle");
  constexpr std::uint64_t offset = 4096 + 90;
  constexpr std::uint32_t rows = 4099;
  const std::array<std::int32_t, 9> prompt{151644, 872,  198, 248044, 9707,
                                           11,     1917, 0,   151645};
  std::mt19937 random(3917);
  const auto config = FixtureConfig();
  for (const Type type : {Type::kIQ4_NL, Type::kBF16}) {
    const auto path =
        temporary.path() / (type == Type::kIQ4_NL ? "iq4" : "bf16");
    {
      std::ofstream file(path, std::ios::binary);
      file.seekp(offset);
      constexpr std::array<std::uint16_t, 8> scales{
          0, 0x8000, 1, 1023, 0x3C00, 0xBC00, 0x7BFF, 0x3555};
      for (std::uint32_t row = 0; row < rows; ++row) {
        if (type == Type::kIQ4_NL) {
          for (unsigned block = 0; block < 5; ++block) {
            const auto scale = scales[(row + block) % scales.size()];
            file.write(reinterpret_cast<const char*>(&scale), 2);
            for (unsigned col = 0; col < 16; ++col)
              file.put(static_cast<char>(random() & 255U));
          }
        } else {
          for (unsigned col = 0; col < 160; ++col) {
            // Includes zeros, subnormals, infinities and NaN payloads.
            const auto bits = static_cast<std::uint16_t>(random());
            file.write(reinterpret_cast<const char*>(&bits), 2);
          }
        }
      }
      Require(file.good(), "fixture write failed");
    }
    auto table = Open(path, offset, rows, type);
    for (const std::size_t count : {1U, 2U, 8U, 9U}) {
      std::vector<std::uint32_t> ids(count);
      for (auto& id : ids)
        id = random() % rows;
      if (count > 1)
        ids.back() = ids.front();
      Compare(*table, ids);
      CompareChain(*table, config, std::span(prompt).first(count));
      std::array<std::int32_t, 9> random_tokens{};
      for (auto& token : random_tokens)
        token = static_cast<std::int32_t>(random() % 248000);
      CompareChain(*table, config, std::span(random_tokens).first(count));
    }
    const std::array<std::uint32_t, 2> ids{rows - 1, rows - 2};
    auto pending = table->ReadAhead(ids);
    table.reset();
    Require(pending->Wait(), "table destruction did not drain raw jobs");
    auto truncated = Open(path, offset, rows + 1, type);
    const std::array<std::uint32_t, 1> missing{rows};
    auto failed = truncated->ReadAhead(missing);
    Require(failed && !failed->Wait(), "truncated raw row did not fail");
    Compare(*truncated, ids);
  }
}

void CpuPolicy() {
  // Independent scalar table for C_d=30.39+6.88*d, including a late cliff.
  struct Case {
    std::array<float, 7> p;
    unsigned depth;
  };
  const std::array<Case, 8> cases{{
      {{{0, 0, 0, 0, 0, 0, 0}}, 0},
      {{{0.2F, 0.2F, 0.2F, 0.2F, 0.2F, 0.2F, 0.2F}}, 0},
      {{{0.5F, 0.5F, 0.5F, 0.5F, 0.5F, 0.5F, 0.5F}}, 1},
      {{{0.75F, 0.75F, 0.75F, 0.75F, 0.75F, 0.75F, 0.75F}}, 3},
      {{{0.8F, 0.8F, 0.8F, 0.8F, 0.8F, 0.8F, 0.8F}}, 4},
      {{{0.9F, 0.9F, 0.9F, 0.9F, 0.9F, 0.9F, 0.9F}}, 6},
      {{{1, 1, 1, 1, 1, 1, 1}}, 7},
      {{{0.99F, 0.99F, 0.1F, 1, 1, 1, 1}}, 2},
  }};
  for (const auto& test : cases) {
    q::MtpLengthController controller(7, 1, q::DraftPolicy::kFit);
    auto state = controller.State();
    for (unsigned i = 0; i < 7; ++i) {
      state.successes[i] = test.p[i] * 10;
      state.failures[i] = (1 - test.p[i]) * 10;
    }
    Require(controller.Restore(state), "fit state restore failed");
    for (unsigned cap = 0; cap <= 9; ++cap)
      Require(controller.Choose(cap) == std::min(test.depth, cap),
              "fit depth differs from scalar table");
  }
  q::DraftRoundWindow window;
  for (unsigned i = 0; i < 32; ++i)
    Require(!window.Observe(i < 24 ? 2 : 1, 5), "0.35 boundary switched off");
  Require(window.Observe(1, 5), "acceptance below 0.35 did not switch off");
  {
    q::MtpLengthController fixed(7, 1, q::DraftPolicy::kFixed4);
    Require(fixed.Choose(7) == 4 && fixed.Choose(3) == 3, "fixed depth cap");
    fixed.Observe(0, 4);
    Require(fixed.Choose(7) == 4, "fixed depth followed acceptance");
    q::DecodeCycleOptions parsed;
    std::string why;
    Require(q::ParseDecodeCycleOptions(nullptr, "fixed4", nullptr, "1", nullptr,
                                       &parsed, &why) &&
                parsed.policy == q::DraftPolicy::kFixed4,
            "fixed depth parse");
    Require(!q::ParseDecodeCycleOptions(nullptr, "fixed4", nullptr, nullptr,
                                        nullptr, &parsed, &why),
            "fixed depth without debug hooks accepted");
  }
  {
    // A fully accepted chain probes one position deeper, then waits 16 such rounds.
    q::MtpLengthController explore(7, 1, q::DraftPolicy::kFit);
    explore.Observe(3, 3);
    Require(explore.State().probe_depth == 4 && explore.Choose(7) >= 4,
            "fit did not probe a deeper draft");
    explore.Observe(3, 3);
    Require(explore.State().probe_depth == 0, "fit probed again at once");
    explore.Observe(1, 3);
    Require(explore.State().probe_depth == 0, "fit probed after a rejection");
    q::MtpLengthController other(7, 1, q::DraftPolicy::kSerial64);
    other.Observe(3, 3);
    Require(other.State().probe_depth == 0, "serial policy gained a probe");
    q::MtpLengthController quiet(7, 1, q::DraftPolicy::kFitNoProbe);
    quiet.Observe(3, 3);
    Require(quiet.State().probe_depth == 0, "fitnoprobe probed");
  }
  q::MtpLengthController adaptive(7, 1, q::DraftPolicy::kSerial64);
  for (unsigned round = 0; round < 32; ++round) {
    for (unsigned serial = 0; serial < 16 && adaptive.Choose(7) == 0; ++serial)
      adaptive.ObserveArToken();
    const auto depth = adaptive.Choose(7);
    Require(depth != 0, "fit backoff never returned to a probe");
    adaptive.Observe(0, depth);
    Require(adaptive.State().cycle.serial_left == (round == 31 ? 64 : 0),
            "adaptive interval did not begin on the 32nd failed round");
  }
  q::MtpLengthController restored(7, 1, q::DraftPolicy::kSerial64);
  Require(restored.Restore(adaptive.State()), "adaptive state restore failed");
  for (unsigned token = 0; token < 64; ++token) {
    Require(adaptive.Choose(7) == 0 && restored.Choose(7) == 0,
            "serial interval ended before 64 tokens");
    if (token == 19) {
      adaptive.ResetForHead();
      restored.ResetForHead();
    }
    adaptive.ObserveArToken();
    restored.ObserveArToken();
  }
  Require(adaptive.Choose(7) == 1 && restored.Choose(7) == 1 &&
              adaptive.Choose(0) == 0,
          "64 serial tokens did not end in a bounded probe");
  q::MtpLengthController off(7, 1, q::DraftPolicy::kOff);
  Require(off.Choose(7) == 0, "off policy drafted");
  q::MtpLengthController censored(7, 1, q::DraftPolicy::kFit);
  const auto prior = censored.State();
  censored.Observe(1, 7);
  Require(censored.State().successes[2] == prior.successes[2] &&
              censored.State().failures[2] == prior.failures[2],
          "unverified tail became evidence");

  q::DecodeCycleOptions options;
  std::string error;
  Require(options.hc_inject_bf16_fused, "BF16 inject fusion default is off");
  for (const char* value :
       std::array<const char*, 7>{nullptr, "0", "1", "", "true", "2", " 1"}) {
    const bool valid = value == nullptr || std::string_view(value) == "0" ||
                       std::string_view(value) == "1";
    error.clear();
    Require(q::ParseDecodeCycleOptions(nullptr, nullptr, nullptr, nullptr,
                                       nullptr, &options, &error, nullptr,
                                       nullptr, nullptr, value) == valid,
            "BF16 inject fusion literal parsing failed");
    if (valid)
      Require(options.hc_inject_bf16_fused ==
                  (value == nullptr || std::string_view(value) == "1"),
              "BF16 inject fusion value failed");
    else
      Require(error == "GUFO_HC_INJECT_BF16_FUSED must be 0 or 1",
              "BF16 inject fusion error failed");
  }
  options = {};
  bool dump = true;
  Require(q::ParseLogitEvalDump(nullptr, nullptr, &dump, &error) && !dump &&
              q::ParseLogitEvalDump("0", nullptr, &dump, &error) && !dump &&
              !q::ParseLogitEvalDump("1", nullptr, &dump, &error) &&
              !q::ParseLogitEvalDump("1", "true", &dump, &error) &&
              !q::ParseLogitEvalDump("yes", "1", &dump, &error) &&
              q::ParseLogitEvalDump("1", "1", &dump, &error) && dump &&
              q::ParseLogitEvalDump("heads", "1", &dump, &error) && dump,
          "head dump debug gate");
  Require(options.ple_readahead && options.policy == q::DraftPolicy::kDefault,
          "option defaults do not enable read-ahead and the current policy");
  Require(q::ParseDecodeCycleOptions(nullptr, nullptr, nullptr, nullptr,
                                     nullptr, &options, &error) &&
              options.policy == q::DraftPolicy::kDefault &&
              options.ple_readahead,
          "unset knobs do not enable read-ahead and the current policy");
  struct PolicyCase {
    const char* value;
    q::DraftPolicy policy;
  };
  const std::array<PolicyCase, 4> policies{{
      {"current", q::DraftPolicy::kDefault},
      {"fit", q::DraftPolicy::kFit},
      {"off", q::DraftPolicy::kOff},
      {"serial64", q::DraftPolicy::kSerial64},
  }};
  for (const char* ahead : {"0", "1"})
    for (const auto& policy : policies)
      Require(q::ParseDecodeCycleOptions(ahead, policy.value, nullptr, nullptr,
                                         nullptr, &options, &error) &&
                  options.ple_readahead == (std::string_view(ahead) == "1") &&
                  options.policy == policy.policy,
              "explicit decode-cycle knobs select the wrong path");
  Require(!q::ParseDecodeCycleOptions("true", nullptr, nullptr, nullptr,
                                      nullptr, &options, &error) &&
              !q::ParseDecodeCycleOptions("", nullptr, nullptr, nullptr,
                                          nullptr, &options, &error) &&
              !q::ParseDecodeCycleOptions(nullptr, "Fit", nullptr, nullptr,
                                          nullptr, &options, &error) &&
              !q::ParseDecodeCycleOptions(nullptr, "", nullptr, nullptr,
                                          nullptr, &options, &error) &&
              !q::ParseDecodeCycleOptions(nullptr, "default", nullptr, nullptr,
                                          nullptr, &options, &error) &&
              !q::ParseDecodeCycleOptions(nullptr, nullptr, "wrong", "0",
                                          nullptr, &options, &error) &&
              !q::ParseDecodeCycleOptions(nullptr, nullptr, "shift", "1", "0",
                                          &options, &error),
          "invalid knobs were not refused");
  for (const char* mode : {"wrong", "shift", "repeat"})
    Require(q::ParseDecodeCycleOptions("1", "serial64", mode, "1", "32",
                                       &options, &error),
            "valid knobs refused");
  for (unsigned id = 0; id < 100; ++id) {
    Require(q::FalsifyDraftToken(q::DraftFalsify::kWrong, id, 17, 100, 9, id) !=
                static_cast<std::int32_t>(id),
            "wrong falsifier kept original");
    Require(q::FalsifyDraftToken(q::DraftFalsify::kShift, id, 17, 100, 9, id) ==
                static_cast<std::int32_t>((id + 1) % 100),
            "shift falsifier");
    Require(q::FalsifyDraftToken(q::DraftFalsify::kRepeat, id, 17, 100, 9,
                                 id) == 17,
            "repeat falsifier");
  }
}

void RealRows(q::test::RealPleTable& real) {
  std::mt19937 random(711);
  std::vector<std::uint32_t> ids(512);
  for (auto& id : ids)
    id = random() % real.table->Rows();
  Compare(*real.table, ids);
  std::vector<std::uint32_t> prompt_rows(real.prompt.size() *
                                         real.config.ple_heads);
  q::NgramHistory history;
  q::HashNgramRows(real.config, history, real.prompt, prompt_rows);
  Compare(*real.table, prompt_rows);
  for (const std::size_t count : {1U, 2U, 8U, 9U})
    CompareChain(*real.table, real.config, std::span(real.prompt).first(count));
  std::printf(
      "PASS real byte comparison: %zu random/prompt rows; token counts "
      "1,2,8,9\n",
      ids.size() + prompt_rows.size());
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc == 1) {
      CpuRows();
      CpuPolicy();
      std::puts(
          "PASS CPU reference: raw rows and fit scalar table; serial64 and "
          "strict debug knobs");
    } else if ((argc == 2 || argc == 3) &&
               std::string_view(argv[1]) == "--real") {
      const char* path =
          argc == 3 ? argv[2] : std::getenv("GUFO_PLE_TEST_MODEL");
      if (!path || !path[0]) {
        // Needs a real model: ctest reports SKIP (SKIP_RETURN_CODE 77) instead of a red.
        std::puts("SKIP: set GUFO_PLE_TEST_MODEL to the first production shard");
        return 77;
      }
      auto real = q::test::OpenRealPle(std::filesystem::u8path(path));
      RealRows(real);
    } else {
      std::puts(
          "usage: decode_cycle_test [--real first-production-shard.gguf]");
      return 2;
    }
  } catch (const std::exception& error) {
    std::fprintf(stderr, "FAIL: %s\n", error.what());
    return 1;
  }
  return 0;
}
