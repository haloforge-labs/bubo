// L-REL-6: LegacyLogprobs for a long scored prompt stays inside a memory cap,
// and a response above the token cap is refused with a 413 error. CPU only.

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <new>
#include <optional>
#include <string>
#include <vector>

#include "src/cli/serve/logprobs.hpp"

namespace {

std::atomic<std::size_t> g_live{0};
std::atomic<std::size_t> g_peak{0};

constexpr std::size_t kHeader = 16;

void* Allocate(std::size_t size) {
  auto* base = static_cast<unsigned char*>(std::malloc(size + kHeader));
  if (base == nullptr)
    throw std::bad_alloc();
  *reinterpret_cast<std::size_t*>(base) = size;
  const auto live = g_live.fetch_add(size) + size;
  auto peak = g_peak.load();
  while (live > peak && !g_peak.compare_exchange_weak(peak, live)) {
  }
  return base + kHeader;
}

void Release(void* p) noexcept {
  if (p == nullptr)
    return;
  auto* base = static_cast<unsigned char*>(p) - kHeader;
  g_live.fetch_sub(*reinterpret_cast<std::size_t*>(base));
  std::free(base);
}

}  // namespace

void* operator new(std::size_t size) { return Allocate(size); }
void operator delete(void* p) noexcept { Release(p); }
void operator delete(void* p, std::size_t) noexcept { Release(p); }

namespace {

void Expect(bool ok, const char* what) {
  if (!ok) {
    std::cerr << "FAIL: " << what << "\n";
    std::exit(1);
  }
}

void TestLargeResponseMemory() {
  using namespace gufo;
  constexpr std::size_t kTokens = 30000;
  constexpr std::size_t kTop = 5;
  std::vector<sampling::TokenId> tokens(kTokens);
  std::vector<std::optional<sampling::TokenLogprob>> scores(kTokens);
  for (std::size_t i = 0; i < kTokens; ++i) {
    tokens[i] = static_cast<sampling::TokenId>(i % 997);
    if (i == 0)
      continue;
    sampling::TokenLogprob score{tokens[i], -1.0 - double(i % 7), {}};
    for (std::size_t k = 0; k < kTop; ++k)
      score.top_logprobs.push_back(
          {static_cast<sampling::TokenId>((i + k) % 997), -0.5 - double(k)});
    scores[i] = std::move(score);
  }
  const server::LogprobTokenText text = [](sampling::TokenId token) {
    return "t" + std::to_string(token);
  };
  const auto baseline = g_live.load();
  g_peak = baseline;
  {
    const auto result = server::LegacyLogprobs(tokens, scores, text);
    const auto retained = g_live.load() - baseline;
    const auto peak = g_peak.load() - baseline;
    std::cout << "LegacyLogprobs " << kTokens << " tokens x " << kTop
              << " top: retained " << (retained >> 10) << " KiB, peak "
              << (peak >> 10) << " KiB\n";
    Expect(result.find("tokens")->items().size() == kTokens,
           "all tokens are present");
    Expect(peak <= (std::size_t{256} << 20), "peak stays under 256 MiB");
    // Reserved arrays do not double while growing: the peak is the result
    // plus a small transient, not 1.5 to 2 times the result.
    Expect(peak <= retained + retained / 8 + (std::size_t{1} << 20),
           "peak is close to the retained size");
  }
}

void TestResponseCap() {
  using namespace gufo;
  const std::size_t over = server::kMaxPromptScoreResponseTokens + 1;
  std::vector<sampling::TokenId> tokens(over, 1);
  std::vector<std::optional<sampling::TokenLogprob>> scores(over);
  const server::LogprobTokenText text = [](sampling::TokenId) {
    return std::string("x");
  };
  bool refused = false;
  try {
    (void)server::LegacyLogprobs(tokens, scores, text);
  } catch (const server::TextGenerationError& error) {
    refused = error.http_status() == 413 &&
              std::string(error.stable_code()) == "response_too_large" &&
              std::string(error.what()).find("65537") != std::string::npos &&
              std::string(error.what()).find("65536") != std::string::npos;
  }
  Expect(refused, "above the cap LegacyLogprobs fails with 413 and both numbers");
  tokens.resize(server::kMaxPromptScoreResponseTokens);
  scores.resize(server::kMaxPromptScoreResponseTokens);
  (void)server::LegacyLogprobs(tokens, scores, text);
}

void TestScoringHelpers() {
  using namespace gufo::server;
  constexpr std::uint64_t mib = std::uint64_t{1} << 20;
  Expect(!PromptScoreHostShortfall(1000, 100 * mib, 50 * mib, 4096 * mib),
         "enough host memory passes");
  const auto shortfall =
      PromptScoreHostShortfall(25800, 3000 * mib, 500 * mib, 2000 * mib);
  Expect(shortfall && shortfall->find("25800 tokens") != std::string::npos &&
             shortfall->find("only 2000 MiB is available") != std::string::npos,
         "a shortfall names the tokens and the available memory");
  bool converted = false;
  try {
    GuardPromptScoringStage("teacher forcing", 4321,
                            [] { throw std::bad_alloc(); });
  } catch (const TextGenerationError& error) {
    const std::string what = error.what();
    converted = error.http_status() == 503 &&
                what.find("teacher forcing") != std::string::npos &&
                what.find("token 4321") != std::string::npos &&
                what.find("bad alloc") == std::string::npos;
  }
  Expect(converted, "bad_alloc becomes a stage and token message (503)");
  Expect(GuardPromptScoringStage("x", 0, [] { return 7; }) == 7,
         "the guard passes the result through");
  bool limit_400 = false;
  try {
    CheckPromptScoreLimits(3000, 2048);
  } catch (const std::invalid_argument& error) {
    const std::string what = error.what();
    limit_400 = what.find("3000") != std::string::npos &&
                what.find("2048") != std::string::npos;
  }
  Expect(limit_400, "over --max-score-tokens names both numbers");
  CheckPromptScoreLimits(2048, 2048);
}

}  // namespace

int main() {
  TestScoringHelpers();
  TestResponseCap();
  TestLargeResponseMemory();
  std::cout << "legacy logprobs memory tests passed\n";
  return 0;
}
