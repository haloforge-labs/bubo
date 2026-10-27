#include "src/models/diffusion_gemma/read.hpp"

#include <cmath>
#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "src/cli/serve/diffusion_gemma_read.hpp"
#include "src/models/diffusion_gemma/cache_state.hpp"

namespace dg = gufo::models::diffusion_gemma;
namespace server = gufo::server;

namespace {

int failures = 0;
void Expect(bool condition, const char* message) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", message);
    ++failures;
  }
}
template<typename Function>
void Rejects(Function&& function, const char* fragment) {
  try {
    function();
    Expect(false, fragment);
  } catch (const std::invalid_argument& error) {
    Expect(std::string(error.what()).find(fragment) != std::string::npos,
           fragment);
  }
}

void PlanningAndPins() {
  const auto first = dg::PlanReadStep(0, 1);
  Expect(!first.self_condition && !first.make_probabilities && first.finish,
         "one-step read skips next-step probabilities and finishes");
  const auto two_first = dg::PlanReadStep(0, 2);
  const auto two_last = dg::PlanReadStep(1, 2);
  Expect(!two_first.self_condition && two_first.make_probabilities &&
             !two_first.finish,
         "first of two steps prepares probabilities");
  Expect(two_last.self_condition && !two_last.make_probabilities &&
             two_last.finish,
         "later step uses self-conditioning and reaches cap");
  Rejects([] { (void)dg::PlanReadStep(2, 2); }, "invalid diffusion read step");

  std::vector<std::int32_t> seed = {1, 2, 3, 4};
  auto current = seed;
  const std::vector<std::uint8_t> pins = {1, 0, 1, 0};
  const std::vector<std::uint8_t> accepted = {0, 1, 0, 0};
  const std::vector<std::int32_t> sampled = {9, 8, 7, 6};
  dg::AdvanceReadCanvas(current, seed, pins, accepted, sampled,
                        [] { return 12; });
  Expect(current == std::vector<std::int32_t>({1, 8, 3, 12}),
         "pins hold seed; free positions accept or renoise");
}

void Scores() {
  std::vector<float> logits(32, -3.0F);
  logits[4] = 9.0F;
  logits[7] = 9.0F;
  logits[25] = 2.0F;
  const std::vector<std::int32_t> labels = {4, 25, 25, 31};
  const auto row = dg::ScoreLogitsRow(logits, 30.0F, 20, labels);
  Expect(row.token == 4 && row.top_logprobs.size() == 20,
         "tie chooses lower token ID and returns top 20");
  Expect(row.top_logprobs[0].token == 4 && row.top_logprobs[1].token == 7,
         "top candidates are ordered including ties");
  Expect(row.label_logprobs.size() == 1 && row.label_logprobs[0].token == 31,
         "inside top and duplicate labels are omitted");
  double sum = 0;
  for (const float raw : logits)
    sum += std::exp(30.0 * std::tanh(raw / 30.0));
  const double expected = 30.0 * std::tanh(9.0 / 30.0) - std::log(sum);
  Expect(std::abs(row.logprob - expected) < 1e-5,
         "score is full-vocabulary temperature-1 log-softmax");
  const std::vector<std::int32_t> outside = {31, 31};
  const auto small = dg::ScoreLogitsRow(logits, 30.0F, 2, outside);
  Expect(
      small.label_logprobs.size() == 1 && small.label_logprobs[0].token == 31,
      "exact outside label and duplicate handling");
}

void BackendContract() {
  server::DiffusionReadParams params;
  params.canvas_length = 255;
  params.read_only = true;
  Rejects(
      [&] {
        (void)server::PlanDiffusionGemmaRead(params, 256, 262144, 20, 0, true);
      },
      "not supported yet");
  params.canvas_length = 256;
  params.constrained = true;
  Rejects(
      [&] {
        (void)server::PlanDiffusionGemmaRead(params, 256, 262144, 20, 0, true);
      },
      "diffusion_constrained is not supported yet");
  params.constrained = false;
  params.read_only = false;
  Rejects(
      [&] {
        (void)server::PlanDiffusionGemmaRead(params, 256, 262144, 20, 0, true);
      },
      "diffusion_read_only");
  params.read_only = true;
  params.max_steps = 9;
  Rejects(
      [&] {
        (void)server::PlanDiffusionGemmaRead(params, 256, 262144, 20, 0, true);
      },
      "supports 1..8");
  params.max_steps = 1;
  params.seed_canvas = std::vector<gufo::sampling::TokenId>(255, 1);
  Rejects(
      [&] {
        (void)server::PlanDiffusionGemmaRead(params, 256, 262144, 20, 0, true);
      },
      "exactly 256");
  params.seed_canvas->push_back(1);
  params.pinned_positions = {3};
  params.max_steps = 2;
  const auto plan =
      server::PlanDiffusionGemmaRead(params, 256, 262144, 20, 123, true);
  Expect(plan.seed_canvas.size() == 256 && plan.pinned_positions[0] == 3 &&
             plan.max_steps == 2 && plan.seed == 123,
         "serve contract maps seeded read to engine options");

  dg::ReadResult scripted;
  scripted.stats.prompt_tokens = 47;
  scripted.stats.reused_tokens = 19;
  scripted.rows.resize(256);
  for (int i = 0; i < 256; ++i) {
    scripted.rows[i].token = 7;
    scripted.rows[i].logprob = -0.25;
    scripted.rows[i].top_logprobs = {{7, -0.25}, {8, -1.25}};
    scripted.rows[i].label_logprobs = {{31, -3.0}};
  }
  const auto result = server::DiffusionGemmaReadResult(scripted, "decision");
  Expect(result.prompt_tokens == 47 && result.cached_prompt_tokens == 19 &&
             result.prefill_tokens == 28,
         "scripted engine result retains actual prompt and reused counts");
  Expect(result.diffusion_rows && result.diffusion_rows->size() == 256 &&
             (*result.diffusion_rows)[0].top_logprobs[1].token == 8 &&
             (*result.diffusion_rows)[255].label_logprobs[0].token == 31,
         "scripted engine rows reach serve result intact");
}

void ReadCacheBoundary() {
  dg::CacheState cache(8, 4);
  std::vector<std::int32_t> prompt(12);
  std::iota(prompt.begin(), prompt.end(), 1);
  (void)cache.Append(prompt);
  for (int step = 0; step < 2; ++step) {
    const auto plan = dg::PlanReadStep(step, 2);
    Expect(cache.length() == static_cast<int>(prompt.size()),
           "read step does not append canvas to cache");
    if (plan.finish)
      Expect(cache.ReusableLength(prompt) == static_cast<int>(prompt.size()),
             "prompt remains reusable after read step cap");
  }
}

}  // namespace

int main() {
  PlanningAndPins();
  Scores();
  BackendContract();
  ReadCacheBoundary();
  if (!failures)
    std::printf("diffusion_gemma_read_test: OK\n");
  return failures ? 1 : 0;
}
