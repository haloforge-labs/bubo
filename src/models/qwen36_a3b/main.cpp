// gufo-a3b: Qwen3.6-35B-A3B exploration driver. Loads the GGUF, renders one
// Qwen ChatML turn, prefills it and streams a sampled reply with timings.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "src/core/sampling.hpp"
#include "src/models/qwen/tokenizer.hpp"
#include "src/models/qwen36_a3b/attention.hpp"
#include "src/models/qwen36_a3b/engine.hpp"
#include "src/models/qwen36_a3b/prompt_lookup.hpp"

namespace {

using Clock = std::chrono::steady_clock;
double Ms(Clock::time_point a, Clock::time_point b) {
  return std::chrono::duration<double, std::milli>(b - a).count();
}

/// Environment defaults: GUFO_A3B_MODEL (--model) and GUFO_A3B_DATA, the
/// directory of the bench inputs (bench-corpus.txt and the lookup bench's
/// source files; default: the current directory).
std::string EnvOr(const char* name, const char* otherwise) {
  const char* value = std::getenv(name);
  return value != nullptr && *value != '\0' ? value : otherwise;
}

std::string DataFile(const std::string& name) {
  return EnvOr("GUFO_A3B_DATA", ".") + "/" + name;
}

void Usage() {
  std::fprintf(stderr,
               "usage: gufo-a3b [--model PATH] [--prompt TEXT] [-n TOKENS]\n"
               "                [--temp T] [--top-k K] [--top-p P] [--seed S]\n"
               "                [--no-think] [--raw] [--ctx N] [--chunk 1-8]\n"
               "                [--ppl FILE] [--ppl-ctx N] [--fn-gate] "
               "[--no-graph] [--legacy] [--profile]\n"
               "                [--mtp DRAFTS] (speculative path; 0 = its "
               "plain decode)\n"
               "                [--bench probe|depth] [--seeds N] [--depths "
               "A,B,..] [--corpus FILE]\n"
               "                [--draft-vocab FILE] [--survival F] "
               "[--draft-sample] [--gpu-chain]\n"
               "                [--prefill-chunk N] (0 = 8-row decode kernels) "
               "[--fn-attention]\n"
               "                [--lookup MIN_MATCH] [--lookup-after-mtp] "
               "[--bench lookup] [--coupled]\n"
               "                [--dflash DRAFT.gguf] [--dflash-n DRAFTS]\n"
               "(serving: gufo serve --model A3B.gguf; see "
               "docs/models/qwen3.6-35b-a3b)\n");
}

}  // namespace

int main(int argc, char** argv) {
  using namespace gufo;
  std::string model = EnvOr("GUFO_A3B_MODEL", "");
  std::string prompt = "Write a short story about a lighthouse keeper.";
  int n_predict = 256;
  sampling::SamplingConfig sc;
  sc.temperature = 1.0F;
  sc.top_k = 20;
  sc.top_p = 0.95F;
  sc.seed = 1234;
  bool think = true;
  bool raw = false;
  std::size_t chunk = models::qwen36_a3b::Engine::kMaxRows;
  std::string ppl_file;
  std::size_t ppl_ctx = 0;
  bool ppl_wide = false;  // --ppl-ctx chunks through the wide prefill
  // --ppl-deep D: prefill D corpus tokens through the 8-row path and the wide
  // path, then score the next --ppl-score tokens with the decode kernels.
  std::size_t ppl_deep = 0;
  std::size_t ppl_score = 1024;
  // --ppl-deep-noise: the second pass is the 8-row path again with only the
  // decode attention's summation order changed (Flash-Next's kernel): the
  // KL floor of harmless rounding differences at that depth.
  bool ppl_noise = false;
  // --ppl-ref FILE: the reference pass's logits are loaded from FILE when it
  // holds them (same D and S), else computed and saved there.
  std::string ppl_ref;
  int ablate = 0;
  int mtp = -1;
  // Speculative policy: survival floor (0 = always draft --mtp tokens),
  // sampled instead of greedy proposals, or the fixed on-GPU greedy chain.
  double survival = 0.0;
  bool draft_sample = false;
  bool gpu_chain = false;
  // Prompt lookup: copied proposals after a context match of at least
  // `lookup` tokens (0 = off). A match at the step start skips the MTP
  // draft; --lookup-after-mtp only tries after each kept MTP draft
  // (Flash-Next's order).
  int lookup = 0;
  bool lookup_after_mtp = false;
  // Bench sampler: Gumbel-max with fixed noise, so every drafting policy
  // yields the same text for a seed (see `gumbel` below).
  bool coupled = false;
  // DFlash-2 drafter (a draft GGUF for this target): up to --dflash-n drafts
  // per step (default its maximum, the verify width - 1).
  std::string dflash;
  int dflash_n = 7;
  // --bench lookup: a comma-separated subset of its tasks.
  std::string only;
  // --bench real: the follow-up (prompt checkpoint) turn from this depth up.
  int followup_min = 65536;
  // In-process benches: "probe" (probe-27b.py's three tasks x seeds) and
  // "depth" (longctx-27b.py's corpus slices).
  std::string bench;
  int seeds = 1;
  std::string depths = "1000,8000,16000,32000";
  std::string corpus_file = DataFile("bench-corpus.txt");
  bool ctx_set = false;
  bool depths_set = false;
  bool n_set = false;
  models::qwen36_a3b::Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const auto next = [&]() -> const char* {
      if (i + 1 >= argc) {
        Usage();
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--model")
      model = next();
    else if (a == "--prompt")
      prompt = next();
    else if (a == "-n")
      n_predict = std::atoi(next()), n_set = true;
    else if (a == "--temp")
      sc.temperature = static_cast<float>(std::atof(next()));
    else if (a == "--top-k")
      sc.top_k = std::atoi(next());
    else if (a == "--top-p")
      sc.top_p = static_cast<float>(std::atof(next()));
    else if (a == "--seed")
      sc.seed = std::atoll(next());
    else if (a == "--ctx")
      options.max_context = static_cast<std::uint32_t>(std::atoi(next())),
      ctx_set = true;
    else if (a == "--chunk")
      chunk = std::clamp<std::size_t>(std::atoi(next()), 1,
                                      models::qwen36_a3b::Engine::kMaxRows);
    else if (a == "--ppl")
      ppl_file = next();
    else if (a == "--fn-gate")
      options.fn_gate = true;
    else if (a == "--fn-router")
      options.fn_router = true;
    else if (a == "--fn-attention")
      options.fn_attention = true;
    else if (a == "--ppl-wide")
      ppl_wide = true;
    else if (a == "--ppl-deep")
      ppl_deep = static_cast<std::size_t>(std::atoll(next()));
    else if (a == "--ppl-deep-noise")
      ppl_noise = true;
    else if (a == "--ppl-ref")
      ppl_ref = next();
    else if (a == "--ppl-score")
      ppl_score = static_cast<std::size_t>(std::atoll(next()));
    else if (a == "--prefill-chunk")
      options.prefill_chunk =
          static_cast<std::uint32_t>(std::max(0, std::atoi(next())));
    else if (a == "--no-graph")
      options.graphs = false;
    else if (a == "--legacy")
      options.fused = false;
    else if (a == "--mtp")
      mtp = std::atoi(next());
    else if (a == "--spec-profile")
      options.spec_profile = true;
    else if (a == "--draft-vocab")
      options.draft_vocab = next();
    else if (a == "--survival")
      survival = std::atof(next());
    else if (a == "--draft-sample")
      draft_sample = true;
    else if (a == "--gpu-chain")
      gpu_chain = true;
    else if (a == "--coupled")
      coupled = true;
    else if (a == "--only")
      only = std::string(",") + next() + ",";
    else if (a == "--followup-min")
      followup_min = std::atoi(next());
    else if (a == "--lookup")
      lookup = std::max(0, std::atoi(next()));
    else if (a == "--lookup-after-mtp")
      lookup_after_mtp = true;
    else if (a == "--dflash")
      dflash = next();
    else if (a == "--dflash-n")
      dflash_n = std::max(0, std::atoi(next()));
    else if (a == "--bench")
      bench = next();
    else if (a == "--seeds")
      seeds = std::max(1, std::atoi(next()));
    else if (a == "--depths")
      depths = next(), depths_set = true;
    else if (a == "--corpus")
      corpus_file = next();
    else if (a == "--fused-parts")
      options.fused_parts = static_cast<std::uint32_t>(std::atoi(next()));
    else if (a == "--profile")
      options.profile = true;
    else if (a == "--profile-sync")
      options.profile = options.profile_sync = true;
    else if (a == "--ppl-ctx")
      ppl_ctx = std::atoi(next());
    else if (a == "--ablate")
      ablate = std::atoi(next());
    else if (a == "--no-think")
      think = false;
    else if (a == "--raw")
      raw = true;
    else {
      Usage();
      return 2;
    }
  }

  if (!bench.empty() && bench != "probe" && bench != "depth" &&
      bench != "lookup" && bench != "rows" && bench != "prefill" &&
      bench != "real" && bench != "attn") {
    Usage();
    return 2;
  }
  if (bench == "attn")
    return models::qwen36_a3b::RunDecodeAttentionBench();
  if (!bench.empty() && mtp < 0)
    mtp = 0;
  if (model.empty()) {
    std::fprintf(stderr,
                 "gufo-a3b: --model PATH (or GUFO_A3B_MODEL) is required\n");
    return 2;
  }
  if (bench == "real") {
    // bench-real.ps1's defaults.
    if (!depths_set)
      depths = "0,4096,16384,32768,65536,131072";
    if (!n_set)
      n_predict = 2048;
  }
  if (ppl_deep > 0 && !ctx_set)
    options.max_context = static_cast<std::uint32_t>(
        (ppl_deep + ppl_score + 64 + 4095) / 4096 * 4096);
  if ((bench == "depth" || bench == "real") && !ctx_set) {
    // Room for the deepest prompt (corpus tokens run ~3.3 chars each).
    int deepest = 0;
    for (std::size_t at = 0; at < depths.size();) {
      const std::size_t comma = depths.find(',', at);
      deepest =
          std::max(deepest, std::atoi(depths.substr(at, comma - at).c_str()));
      at = comma == std::string::npos ? depths.size() : comma + 1;
    }
    // The corpus runs denser than the nominal 3.3 characters per token.
    options.max_context = static_cast<std::uint32_t>(
        (deepest * 2 + n_predict + 4096 + 4095) / 4096 * 4096);
  }
  std::string error;
  const auto t0 = Clock::now();
  auto engine = models::qwen36_a3b::Engine::Load(model, options, &error);
  if (!engine) {
    std::fprintf(stderr, "load failed: %s\n", error.c_str());
    return 1;
  }
  if (!dflash.empty()) {
    if (!engine->LoadDFlash(dflash, &error)) {
      std::fprintf(stderr, "DFlash load failed: %s\n", error.c_str());
      return 1;
    }
    if (mtp < 0)
      mtp = 0;  // the speculative path
  }
  auto tok =
      tokenization::QwenTokenizer::CreateFromGguf(engine->gguf(), &error);
  if (!tok) {
    std::fprintf(stderr, "tokenizer failed: %s\n", error.c_str());
    return 1;
  }
  const auto t1 = Clock::now();
  std::fprintf(stderr, "a3b: loaded in %.1f s\n", Ms(t0, t1) / 1000.0);
  if (!ppl_file.empty()) {
    // Teacher-forced perplexity over a raw text file (no BOS, no template).
    std::FILE* f = std::fopen(ppl_file.c_str(), "rb");
    if (f == nullptr) {
      std::fprintf(stderr, "cannot open %s\n", ppl_file.c_str());
      return 1;
    }
    std::string corpus;
    char buf[65536];
    for (std::size_t got; (got = std::fread(buf, 1, sizeof(buf), f)) > 0;)
      corpus.append(buf, got);
    std::fclose(f);
    tokenization::TokenizerOptions topt;
    topt.parse_special_tokens = false;
    const auto ctoks = tok->Encode(corpus, topt);
    std::vector<std::int32_t> t(ctoks.begin(), ctoks.end());
    if (t.size() + 1 > options.max_context)
      t.resize(options.max_context - 1);
    const std::uint32_t V = engine->config().vocab;
    std::vector<float> rows(static_cast<std::size_t>(chunk) * V);
    double nll = 0;
    std::size_t scored = 0, top1 = 0;
    const auto score = [&](const float* row, std::int32_t target) {
      float mx = row[0];
      std::uint32_t arg = 0;
      for (std::uint32_t v = 1; v < V; ++v)
        if (row[v] > mx)
          mx = row[v], arg = v;
      double sum = 0;
      for (std::uint32_t v = 0; v < V; ++v)
        sum += std::exp(double(row[v]) - mx);
      nll += std::log(sum) + mx - row[target];
      top1 += arg == static_cast<std::uint32_t>(target);
      ++scored;
    };
    const auto q0 = Clock::now();
    if (ppl_deep > 0) {
      // Deep-context A/B of the prefill paths: the same D tokens prefilled by
      // the 8-row decode kernels (reference) and by the wide path, then the
      // next S tokens scored identically (decode kernels, 8 rows at a time),
      // so any difference comes from the prefilled KV and recurrent state.
      const std::size_t D = ppl_deep, S = ppl_score;
      if (t.size() < D + S + 1) {
        std::fprintf(stderr, "corpus has %zu tokens, need %zu\n", t.size(),
                     D + S + 1);
        return 1;
      }
      std::vector<float> ref(S * V), cur(S * V);
      const auto pass = [&](bool wide, float* out) {
        engine->Reset();
        const auto a0 = Clock::now();
        const bool noise = wide && ppl_noise;
        engine->SetFnAttention(noise);
        if (noise)
          wide = false;
        if (wide) {
          engine->SetPrefillChunk(2048);
          models::qwen36_a3b::Candidates unused{};
          if (!engine->SpecPrefill({t.data(), D}, &unused, &error))
            return false;
        } else {
          for (std::size_t off = 0; off < D; off += 8) {
            const std::size_t n = std::min<std::size_t>(8, D - off);
            if (!engine->Forward({t.data() + off, n}, nullptr, &error))
              return false;
          }
          if (!engine->Sync(&error))
            return false;
        }
        const double pre = Ms(a0, Clock::now()) / 1000.0;
        for (std::size_t off = 0; off < S; off += 8) {
          const std::size_t n = std::min<std::size_t>(8, S - off);
          if (!engine->Forward({t.data() + D + off, n}, out + off * V, &error,
                               true))
            return false;
        }
        std::fprintf(stderr,
                     "a3b: %s prefill of %zu tokens %.1f s (%.0f t/s)\n",
                     noise  ? "8-row (FN attention)"
                     : wide ? "wide"
                            : "8-row",
                     D, pre, D / pre);
        return true;
      };
      bool have_ref = false;
      const std::size_t ref_bytes = ref.size() * sizeof(float);
      if (!ppl_ref.empty()) {
        if (std::FILE* rf = std::fopen(ppl_ref.c_str(), "rb")) {
          std::uint64_t hdr[2] = {};
          have_ref = std::fread(hdr, sizeof(hdr), 1, rf) == 1 && hdr[0] == D &&
                     hdr[1] == S &&
                     std::fread(ref.data(), 1, ref_bytes, rf) == ref_bytes;
          std::fclose(rf);
          if (have_ref)
            std::fprintf(stderr, "a3b: reference logits from %s\n",
                         ppl_ref.c_str());
        }
      }
      if (!have_ref) {
        if (!pass(false, ref.data())) {
          std::fprintf(stderr, "deep ppl failed: %s\n", error.c_str());
          return 1;
        }
        if (!ppl_ref.empty()) {
          if (std::FILE* rf = std::fopen(ppl_ref.c_str(), "wb")) {
            const std::uint64_t hdr[2] = {D, S};
            std::fwrite(hdr, sizeof(hdr), 1, rf);
            std::fwrite(ref.data(), 1, ref_bytes, rf);
            std::fclose(rf);
          }
        }
      }
      if (!pass(true, cur.data())) {
        std::fprintf(stderr, "deep ppl failed: %s\n", error.c_str());
        return 1;
      }
      // Row j predicts t[D + j + 1].
      double nll_ref = 0, nll_cur = 0, kl = 0, kl_max = 0;
      std::size_t top_ref = 0, top_cur = 0, agree = 0, n = 0;
      std::vector<double> pr(V), pc(V);
      for (std::size_t j = 0; j + 1 < S; ++j) {
        const float* a = ref.data() + j * V;
        const float* b = cur.data() + j * V;
        const auto softmax = [&](const float* x, std::vector<double>& p,
                                 std::uint32_t* arg) {
          float mx = x[0];
          *arg = 0;
          for (std::uint32_t v = 1; v < V; ++v)
            if (x[v] > mx)
              mx = x[v], *arg = v;
          double sum = 0;
          for (std::uint32_t v = 0; v < V; ++v)
            sum += p[v] = std::exp(double(x[v]) - mx);
          for (std::uint32_t v = 0; v < V; ++v)
            p[v] /= sum;
        };
        std::uint32_t arg_ref = 0, arg_cur = 0;
        softmax(a, pr, &arg_ref);
        softmax(b, pc, &arg_cur);
        const std::int32_t target = t[D + j + 1];
        nll_ref -= std::log(std::max(pr[target], 1e-300));
        nll_cur -= std::log(std::max(pc[target], 1e-300));
        double k = 0;
        for (std::uint32_t v = 0; v < V; ++v)
          if (pr[v] > 0)
            k += pr[v] * (std::log(pr[v]) - std::log(std::max(pc[v], 1e-300)));
        kl += k;
        kl_max = std::max(kl_max, k);
        top_ref += arg_ref == static_cast<std::uint32_t>(target);
        top_cur += arg_cur == static_cast<std::uint32_t>(target);
        agree += arg_ref == arg_cur;
        ++n;
      }
      std::printf(
          "deep ppl @ %zu tokens, next %zu scored:\n"
          "  8-row prefill (reference): PPL %.4f, top-1 %.2f%%\n"
          "  %-26s PPL %.4f, top-1 %.2f%%  (%+.2f%%)\n"
          "  KL(ref || wide) mean %.5f, max %.4f nats; argmax agreement "
          "%.2f%%\n",
          D, S, std::exp(nll_ref / n), 100.0 * top_ref / n,
          ppl_noise ? "8-row, FN attention:" : "wide prefill:",
          std::exp(nll_cur / n), 100.0 * top_cur / n,
          100.0 * (std::exp(nll_cur / n) / std::exp(nll_ref / n) - 1.0), kl / n,
          kl_max, 100.0 * agree / n);
      return 0;
    }
    if (ppl_ctx > 0) {
      // llama-perplexity's scheme: independent ppl_ctx-token chunks, each
      // from a clean state, scoring only positions [ctx/2, ctx-1).
      const std::size_t N = ppl_ctx;
      const std::size_t first = N / 2;
      const std::size_t n_chunk = t.size() / N;
      if (ppl_wide)
        rows.resize(N * V);
      for (std::size_t c = 0; c < n_chunk; ++c) {
        engine->Reset();
        const std::size_t start = c * N;
        if (ppl_wide) {
          // The whole chunk through the wide prefill path.
          if (!engine->WideLogits({t.data() + start, N}, rows.data(), &error)) {
            std::fprintf(stderr, "wide logits failed: %s\n", error.c_str());
            return 1;
          }
          for (std::size_t j = first; j + 1 < N; ++j)
            score(rows.data() + j * V, t[start + j + 1]);
          std::fprintf(stderr, "[%zu]%.4f,", c + 1, std::exp(nll / scored));
          continue;
        }
        for (std::size_t off = 0; off < N; off += chunk) {
          const std::size_t n = std::min(chunk, N - off);
          if (!engine->Forward({t.data() + start + off, n}, rows.data(), &error,
                               true)) {
            std::fprintf(stderr, "ppl forward failed: %s\n", error.c_str());
            return 1;
          }
          for (std::size_t r = 0; r < n; ++r) {
            const std::size_t j = off + r;
            if (j >= first && j + 1 < N)
              score(rows.data() + r * V, t[start + j + 1]);
          }
        }
        std::fprintf(stderr, "[%zu]%.4f,", c + 1, std::exp(nll / scored));
      }
      std::fprintf(stderr,
                   "\na3b: llama-perplexity scheme -c %zu: %zu chunks, "
                   "Final estimate: PPL = %.4f (top-1 %.2f%%) in %.1f s\n",
                   N, n_chunk, std::exp(nll / scored), 100.0 * top1 / scored,
                   Ms(q0, Clock::now()) / 1000.0);
      return 0;
    }
    for (std::size_t off = 0; off + 1 < t.size(); off += chunk) {
      const std::size_t n = std::min(chunk, t.size() - 1 - off);
      if (!engine->Forward({t.data() + off, n}, rows.data(), &error, true)) {
        std::fprintf(stderr, "ppl forward failed: %s\n", error.c_str());
        return 1;
      }
      for (std::size_t r = 0; r < n; ++r)
        score(rows.data() + r * V, t[off + r + 1]);
    }
    const double s = Ms(q0, Clock::now()) / 1000.0;
    std::fprintf(stderr,
                 "a3b: ppl %.6f over %zu positions (top-1 self %.2f%%, "
                 "chunk %zu%s) in %.1f s\n",
                 std::exp(nll / scored), scored, 100.0 * top1 / scored, chunk,
                 options.fn_gate ? ", FN sigmoid gate" : "", s);
    return 0;
  }

  std::string text = prompt;
  if (!raw) {
    text =
        "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
    text += think ? "<think>\n" : "<think>\n\n</think>\n\n";
  }
  const auto ids = tok->Encode(text);
  std::vector<std::int32_t> prompt_ids(ids.begin(), ids.end());
  const auto im_end = tok->FindSpecialToken("<|im_end|>");
  const auto eos = tok->GetEosTokenId();

  if (mtp >= 0) {
    // Speculative path: host sampling over each row's top-64 candidates
    // (exact for top-k <= 64), greedy MTP drafts accepted with probability
    // p(draft) and otherwise resampled from p without the draft.
    using models::qwen36_a3b::Candidates;
    std::mt19937_64 rng(sc.seed);
    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    using Dist = std::vector<std::pair<std::int32_t, double>>;
    const auto dist = [&](const Candidates& c) {
      Dist d;
      if (sc.temperature <= 0.0F) {
        d.push_back({c.ids[0], 1.0});
        return d;
      }
      const int k = sc.top_k > 0 ? std::min<int>(sc.top_k, Candidates::kCount)
                                 : static_cast<int>(Candidates::kCount);
      double sum = 0.0;
      for (int i = 0; i < k; ++i) {
        const double w = std::exp((c.logits[i] - c.logits[0]) / sc.temperature);
        d.push_back({c.ids[i], w});
        sum += w;
      }
      double cum = 0.0;
      std::size_t keep = d.size();
      for (std::size_t i = 0; i < d.size(); ++i) {
        cum += d[i].second / sum;
        if (cum >= sc.top_p) {
          keep = i + 1;
          break;
        }
      }
      d.resize(keep);
      double total = 0.0;
      for (const auto& e : d)
        total += e.second;
      for (auto& e : d)
        e.second /= total;
      return d;
    };
    const auto sample = [&](const Dist& d, std::int32_t exclude) {
      double total = 0.0;
      for (const auto& e : d)
        if (e.first != exclude)
          total += e.second;
      double u = uniform(rng) * total;
      for (const auto& e : d) {
        if (e.first == exclude)
          continue;
        if ((u -= e.second) <= 0.0)
          return e.first;
      }
      for (auto it = d.rbegin(); it != d.rend(); ++it)
        if (it->first != exclude)
          return it->first;
      return d.front().first;
    };
    // --coupled: Gumbel-max sampling with noise fixed per (seed, reply
    // position, token). Each token is still an exact draw from `d`, but the
    // text no longer depends on which drafts were proposed, so policies can
    // be compared on one identical text (and must produce it). A draft is
    // accepted iff it is the position's sample: probability p(d), as usual.
    std::uint64_t coupled_seed = 0;
    const auto gumbel = [&](const Dist& d, std::size_t pos) {
      std::int32_t best = d.front().first;
      double best_key = -1e300;
      for (const auto& e : d) {
        std::uint64_t z =
            coupled_seed ^ (pos * 0x9E3779B97F4A7C15ULL) ^
            (static_cast<std::uint64_t>(e.first) * 0xD1B54A32D192ED03ULL);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z ^= z >> 31;
        const double u = (static_cast<double>(z >> 11) + 0.5) * 0x1.0p-53;
        const double key = std::log(e.second) - std::log(-std::log(u));
        if (key > best_key)
          best_key = key, best = e.first;
      }
      return best;
    };
    const auto is_stop = [&](std::int32_t t) {
      return t == static_cast<std::int32_t>(eos) ||
             (im_end && t == static_cast<std::int32_t>(*im_end));
    };
    struct Result {
      std::size_t prompt_n{0};
      double prefill_ms{0};
      int generated{0};
      double decode_ms{0};
      std::size_t cycles{0}, drafted{0}, accepted{0};
      std::size_t looked{0}, lookup_accepted{0};  // prompt lookup drafts
      std::uint64_t hash{1469598103934665603ULL};
      std::vector<std::int32_t> out;  // the reply's tokens
      std::size_t cached{0};          // prompt tokens served by the checkpoint
      bool stopped{false};            // ended on a stop token (else the budget)
      bool ok{false};
    };
    // Prompt checkpoint for --bench real: kSave prefills the prompt without
    // its assistant suffix, saves the engine checkpoint, then the suffix;
    // kResume restores it when the prompt extends the saved prefix.
    enum CacheMode { kNoCache, kSave, kResume };
    std::vector<std::int32_t> ckpt_tokens;
    const auto suffix_ids = [&] {
      const auto e =
          tok->Encode(std::string("<|im_start|>assistant\n") +
                      (think ? "<think>\n" : "<think>\n\n</think>\n\n"));
      return std::vector<std::int32_t>(e.begin(), e.end());
    }();
    // One request from a clean state: prefill, then draft/verify steps.
    const auto run = [&](const std::vector<std::int32_t>& ids, int budget,
                         std::uint64_t seed, bool echo,
                         CacheMode cache = kNoCache) {
      Result r;
      // Diagnostic: A3B_NO_RESUME prefills follow-ups in full (A/B).
      static const bool no_resume = std::getenv("A3B_NO_RESUME") != nullptr;
      if (cache == kResume && no_resume)
        cache = kNoCache;
      if (cache == kResume && !ckpt_tokens.empty() &&
          ids.size() > ckpt_tokens.size() &&
          std::equal(ckpt_tokens.begin(), ckpt_tokens.end(), ids.begin())) {
        if (!engine->RestoreCheckpoint(&error)) {
          std::fprintf(stderr, "checkpoint restore failed: %s\n",
                       error.c_str());
          return r;
        }
        r.cached = ckpt_tokens.size();
      } else {
        engine->Reset();
      }
      rng.seed(seed);
      coupled_seed = seed * 0x9E3779B97F4A7C15ULL + 0x632BE59BD9B4E019ULL;
      // The context so far (prompt, then every emitted token) and its
      // lookup index.
      std::vector<std::int32_t> hist(ids.begin(), ids.end());
      models::qwen36_a3b::PromptLookup index(static_cast<std::size_t>(lookup));
      const auto emit = [&](std::int32_t t) {
        r.hash = (r.hash ^ static_cast<std::uint32_t>(t)) * 1099511628211ULL;
        ++r.generated;
        hist.push_back(t);
        r.out.push_back(t);
        if (!echo)
          return;
        const std::string piece = tok->DecodeTokenCopy(t);
        std::fwrite(piece.data(), 1, piece.size(), stdout);
        std::fflush(stdout);
      };
      Candidates last{};
      const auto p0 = Clock::now();
      // kSave: the prompt up to its assistant suffix, the checkpoint, then
      // the suffix.
      std::size_t split = r.cached;
      if (cache == kSave && ids.size() > suffix_ids.size() &&
          std::equal(
              suffix_ids.begin(), suffix_ids.end(),
              ids.end() - static_cast<std::ptrdiff_t>(suffix_ids.size()))) {
        split = ids.size() - suffix_ids.size();
        Candidates unused{};
        if (!engine->SpecPrefill({ids.data(), split}, &unused, &error) ||
            !engine->SaveCheckpoint(&error)) {
          std::fprintf(stderr, "prefill failed: %s\n", error.c_str());
          return r;
        }
        ckpt_tokens.assign(ids.begin(),
                           ids.begin() + static_cast<std::ptrdiff_t>(split));
      }
      if (!engine->SpecPrefill({ids.data() + split, ids.size() - split}, &last,
                               &error)) {
        std::fprintf(stderr, "prefill failed: %s\n", error.c_str());
        return r;
      }
      r.prompt_n = ids.size() - r.cached;
      r.prefill_ms = Ms(p0, Clock::now());
      if (echo) {
        std::fprintf(stderr, "a3b: prompt %zu tokens in %.0f ms (%.1f t/s)\n\n",
                     r.prompt_n, r.prefill_ms,
                     r.prompt_n * 1000.0 / r.prefill_ms);
        std::fprintf(stdout, "%s", raw ? "" : "<think>\n");
      }
      std::int32_t x = coupled ? gumbel(dist(last), 0) : sample(dist(last), -1);
      std::vector<std::int32_t> drafts;
      std::vector<Candidates> rows;
      const auto d0 = Clock::now();
      bool stop = is_stop(x);
      while (!stop && r.generated < budget) {
        emit(x);
        const std::uint32_t cap = static_cast<std::uint32_t>(
            std::min<int>(mtp, std::max(0, budget - r.generated)));
        // Proposal distributions of the kept drafts (sampled proposals).
        std::vector<Dist> props;
        // Prompt lookup fills the rest of the step (up to the verify width)
        // with what followed the best match; drafts from `copied` on are
        // copies (point-mass proposals).
        const auto wide = static_cast<std::uint32_t>(
            std::min<int>(models::qwen36_a3b::Engine::kMaxRows - 1,
                          std::max(0, budget - r.generated)));
        std::size_t copied = SIZE_MAX;
        const auto fill = [&] {
          if (lookup == 0 || drafts.size() >= wide)
            return false;
          const auto m = index.Find(hist, drafts);
          if (m.length == 0)
            return false;
          const std::size_t count = std::min<std::size_t>(
              wide - drafts.size(), hist.size() - m.start);
          copied = drafts.size();
          for (std::size_t i = 0; i < count; ++i)
            drafts.push_back(hist[m.start + i]);
          return count != 0;
        };
        drafts.clear();
        if (lookup > 0)
          index.Extend(hist);
        if (engine->HasDFlash() && dflash_n > 0) {
          // DFlash block drafts (top-1 proposals), then prompt lookup copies
          // when the block left room.
          std::vector<float> dprob;
          if (!engine->DFlashDraft(
                  x,
                  std::min<std::uint32_t>(static_cast<std::uint32_t>(dflash_n),
                                          wide),
                  &drafts, &dprob, &error)) {
            std::fprintf(stderr, "\ndflash draft failed: %s\n", error.c_str());
            return r;
          }
          // Survival: verify the block only while the product of the draft
          // head's probabilities stays above the floor (the first draft is
          // always verified).
          if (survival > 0) {
            double alive = 1.0;
            for (std::size_t j = 0; j < drafts.size(); ++j) {
              alive *= dprob[j];
              if (j > 0 && alive < survival) {
                drafts.resize(j);
                break;
              }
            }
          }
          if (lookup > 0)
            (void)fill();
          if (!engine->Verify(x, drafts, &rows, &error)) {
            std::fprintf(stderr, "\nverify failed: %s\n", error.c_str());
            return r;
          }
        } else if (!lookup_after_mtp && fill()) {
          // Copies only: the MTP catch-up still runs inside the verify.
          if (!engine->Verify(x, drafts, &rows, &error, mtp > 0)) {
            std::fprintf(stderr, "\nverify failed: %s\n", error.c_str());
            return r;
          }
        } else if (gpu_chain || cap == 0) {
          // Greedy chain drafted entirely on the GPU, fixed length.
          if (!engine->SpecStep(x, cap, &drafts, &rows, &error)) {
            std::fprintf(stderr, "\nstep failed: %s\n", error.c_str());
            return r;
          }
        } else {
          // Host-driven: one MTP draft at a time; survival stops the chain
          // once the product of the drafts' top probabilities (after the
          // sampler's transform) falls below the floor. The first draft is
          // always verified.
          Candidates q{};
          if (!engine->DraftFirst(x, &q, &error)) {
            std::fprintf(stderr, "\ndraft failed: %s\n", error.c_str());
            return r;
          }
          double alive = 1.0;
          for (;;) {
            Dist qd = dist(q);
            const std::int32_t d = draft_sample ? sample(qd, -1) : q.ids[0];
            alive *= qd.front().second;
            if (!drafts.empty() && alive < survival)
              break;
            drafts.push_back(d);
            props.push_back(std::move(qd));
            if (fill() || drafts.size() >= cap)
              break;
            if (!engine->DraftNext(d, &q, &error)) {
              std::fprintf(stderr, "\ndraft failed: %s\n", error.c_str());
              return r;
            }
          }
          if (!engine->Verify(x, drafts, &rows, &error)) {
            std::fprintf(stderr, "\nverify failed: %s\n", error.c_str());
            return r;
          }
        }
        ++r.cycles;
        r.drafted += drafts.size();
        if (copied < drafts.size())
          r.looked += drafts.size() - copied;
        const auto prob = [](const Dist& d, std::int32_t t) {
          for (const auto& e : d)
            if (e.first == t)
              return e.second;
          return 0.0;
        };
        std::uint32_t keep = 0;
        std::int32_t next = -1;
        for (std::uint32_t j = 0; j < drafts.size(); ++j) {
          const Dist d = dist(rows[j]);
          const bool sampled = !coupled && draft_sample && j < props.size();
          // Greedy proposals: accept with p(d). Sampled: with p(d) / q(d),
          // else draw from the residual max(p - q, 0).
          const double pd = prob(d, drafts[j]);
          const double u = uniform(rng);
          const std::int32_t pick = coupled ? gumbel(d, r.generated) : -1;
          const bool ok = coupled   ? pick == drafts[j]
                          : sampled ? u * prob(props[j], drafts[j]) < pd
                                    : u < pd;
          if (ok) {
            ++r.accepted;
            if (j >= copied)
              ++r.lookup_accepted;
            if (is_stop(drafts[j]) || r.generated >= budget) {
              keep = j + 1;
              stop = true;
              break;
            }
            emit(drafts[j]);
            continue;
          }
          keep = j + 1;
          if (sampled) {
            Dist residual;
            for (const auto& e : d) {
              const double w = e.second - prob(props[j], e.first);
              if (w > 0)
                residual.push_back({e.first, w});
            }
            next = residual.empty() ? sample(d, -1) : sample(residual, -1);
          } else {
            next = coupled ? pick : sample(d, drafts[j]);
          }
          break;
        }
        if (!stop && next < 0) {
          keep = static_cast<std::uint32_t>(drafts.size()) + 1;
          const Dist d = dist(rows[drafts.size()]);
          next = coupled ? gumbel(d, r.generated) : sample(d, -1);
        }
        if (!engine->SpecCommit(keep, &error)) {
          std::fprintf(stderr, "\ncommit failed: %s\n", error.c_str());
          return r;
        }
        x = next;
        stop = stop || is_stop(x);
      }
      r.decode_ms = Ms(d0, Clock::now());
      r.stopped = stop;
      r.ok = true;
      return r;
    };
    const auto render = [&](const std::string& user) {
      std::string t =
          "<|im_start|>user\n" + user + "<|im_end|>\n<|im_start|>assistant\n";
      t += think ? "<think>\n" : "<think>\n\n</think>\n\n";
      const auto e = tok->Encode(t);
      return std::vector<std::int32_t>(e.begin(), e.end());
    };
    const auto line = [&](const char* label, const Result& r) {
      std::printf(
          "%-10s %5d tok  %6.2f t/s  accept %5.1f%%  %.2f tok/cycle  "
          "%6.1f ms/cycle",
          label, r.generated, r.generated * 1000.0 / r.decode_ms,
          r.drafted ? 100.0 * r.accepted / r.drafted : 0.0,
          r.cycles ? static_cast<double>(r.generated) / r.cycles : 0.0,
          r.cycles ? r.decode_ms / r.cycles : 0.0);
      if (lookup > 0)
        std::printf("  lookup %zu/%zu", r.lookup_accepted, r.looked);
      std::printf("  text %012llx\n",
                  static_cast<unsigned long long>(r.hash & 0xffffffffffffULL));
      std::fflush(stdout);
    };
    const auto add = [](Result& sum, const Result& r) {
      sum.generated += r.generated;
      sum.decode_ms += r.decode_ms;
      sum.cycles += r.cycles;
      sum.drafted += r.drafted;
      sum.accepted += r.accepted;
      sum.looked += r.looked;
      sum.lookup_accepted += r.lookup_accepted;
      sum.hash = 0;
    };
    if (!bench.empty()) {
      std::printf(
          "a3b bench: mtp %d (%s%s, survival %.2f%s), lookup %d%s%s, "
          "temp %.2f top-p %.2f top-k %d, thinking %s, max %d tokens\n",
          mtp, gpu_chain ? "gpu chain" : "host drafts",
          draft_sample ? ", sampled drafts" : ", top-1 drafts", survival,
          options.draft_vocab.empty() ? "" : ", draft vocab", lookup,
          lookup > 0 && lookup_after_mtp ? " after mtp" : "",
          coupled ? ", coupled sampler" : "", sc.temperature, sc.top_p,
          sc.top_k, think ? "on" : "off", n_predict);
      if (engine->HasDFlash())
        std::printf("dflash: up to %d drafts per step\n",
                    std::min<int>(dflash_n,
                                  static_cast<int>(engine->DFlashMaxDrafts())));
      // Warm-up (untimed): first-use graph captures are not billed to a run.
      (void)run(render("Say hi."), 16, sc.seed, false);
    }
    if (bench == "probe") {
      // probe-27b.py's three tasks, seeds seed..seed+seeds-1.
      const std::pair<const char*, const char*> tasks[] = {
          {"story",
           "Write a short story about a lighthouse keeper who finds "
           "a message in a bottle."},
          {"code",
           "Write a Python module implementing an LRU cache with TTL "
           "expiry, with type hints and docstrings."},
          {"train",
           "A train leaves at 9:40 and travels 237 km at 83 km/h, "
           "then waits 12 minutes, then travels 118 km at 94 km/h. "
           "When does it arrive? Work it out step by step."}};
      Result all;
      all.decode_ms = 0;
      for (const auto& [name, user] : tasks) {
        Result sum;
        const auto ids = render(user);
        for (int s = 0; s < seeds; ++s) {
          const Result r = run(ids, n_predict, sc.seed + s, false);
          if (!r.ok)
            return 1;
          const std::string label =
              std::string(name) + " s" + std::to_string(sc.seed + s);
          line(label.c_str(), r);
          add(sum, r);
        }
        const std::string label = std::string(name) + " ALL";
        line(label.c_str(), sum);
        add(all, sum);
      }
      line("ALL", all);
      return 0;
    }
    if (bench == "real") {
      // bench-real.ps1 in-process: per depth a summary (mostly thinking) and
      // an HTML card (code) request on fresh corpus slices (cold prefills),
      // and from --followup-min up a follow-up turn of the card conversation
      // that should reuse the prompt checkpoint (cached ~ the whole document).
      std::FILE* f = std::fopen(corpus_file.c_str(), "rb");
      if (f == nullptr) {
        std::fprintf(stderr, "cannot open %s\n", corpus_file.c_str());
        return 1;
      }
      std::string text;
      char buf[65536];
      for (std::size_t got; (got = std::fread(buf, 1, sizeof(buf), f)) > 0;)
        text.append(buf, got);
      std::fclose(f);
      const char* task_names[] = {"summary", "card"};
      const char* tasks[] = {
          "Sum this up for me.",
          "Create the most simple HTML card with a single fact you select from "
          "this text."};
      const char* standalone[] = {
          "Sum up for me how HTTP caching works: what the browser stores, how "
          "it revalidates, and which headers control it.",
          "Create the most simple HTML card with a single interesting fact "
          "about octopuses."};
      const std::string followup =
          "Now make the card dark-themed and add a second fact from the same "
          "text.";
      constexpr std::size_t kCharsPerToken = 4;
      // A byte slice moved off UTF-8 continuation bytes at both ends.
      const auto slice = [&](std::size_t at, std::size_t len) {
        while (at < text.size() && (text[at] & 0xC0) == 0x80)
          ++at;
        std::size_t end = std::min(text.size(), at + len);
        while (end < text.size() && (text[end] & 0xC0) == 0x80)
          ++end;
        return text.substr(at, end - at);
      };
      const auto user_content = [&](int depth, int task, int index) {
        if (depth == 0)
          return std::string(standalone[task]);
        const std::size_t chars =
            static_cast<std::size_t>(depth) * kCharsPerToken;
        const std::size_t span = std::max<std::size_t>(
            1, text.size() > chars ? text.size() - chars : 1);
        const std::size_t offset =
            (static_cast<std::size_t>(task) * 150000 + index * 7000) % span;
        return "Here is a document:\n\n" + slice(offset, chars) + "\n\n" +
               tasks[task];
      };
      const auto encode = [&](const std::string& t) {
        const auto e = tok->Encode(t);
        return std::vector<std::int32_t>(e.begin(), e.end());
      };
      const std::string gen_suffix =
          std::string("<|im_start|>assistant\n") +
          (think ? "<think>\n" : "<think>\n\n</think>\n\n");
      std::string mode = coupled ? "coupled" : "sampled";
      if (mtp > 0)
        mode += "-mtp" + std::to_string(mtp);
      if (!options.draft_vocab.empty())
        mode += "-draftvocab";
      if (survival > 0)
        mode += "-survival";
      if (lookup > 0)
        mode += "-lookup";
      char stamp[32];
      const std::time_t now = std::time(nullptr);
      std::strftime(stamp, sizeof(stamp), "%Y%m%d-%H%M%S",
                    std::localtime(&now));
      const std::string stem = DataFile("bench-real-a3b-") + mode + "-" + stamp;
      std::FILE* replies = std::fopen((stem + ".replies.txt").c_str(), "wb");
      std::FILE* csv = std::fopen((stem + ".csv").c_str(), "wb");
      if (csv != nullptr)
        std::fprintf(csv,
                     "engine,mode,depth,task,prompt_n,cache_n,pp_tps,"
                     "prefill_s,predicted_n,finish,tg_tps,"
                     "draft_accept_pct,wall_s\n");
      std::printf(
          "\n| %7s | %-8s | %10s | %7s | %8s | %9s | %7s | %-6s | %7s | "
          "%8s | %7s |\n",
          "depth", "task", "prompt tok", "cached", "pp t/s", "prefill s",
          "gen tok", "finish", "tg t/s", "accept %", "wall s");
      std::printf(
          "| ------: | :------- | ---------: | ------: | -------: | "
          "--------: | ------: | :----- | ------: | -------: | ------: |\n");
      std::fflush(stdout);
      struct Row {
        int depth;
        std::string task;
        Result r;
      };
      std::vector<Row> table;
      const auto request = [&](int depth, const char* task,
                               const std::vector<std::int32_t>& ids,
                               CacheMode cache, std::string* answer) {
        Result r = run(ids, n_predict, sc.seed, false, cache);
        if (!r.ok)
          return false;
        std::string reply;
        for (const auto id : r.out)
          reply += tok->DecodeTokenCopy(id);
        const std::size_t close = reply.find("</think>");
        std::string content =
            close == std::string::npos ? "" : reply.substr(close + 8);
        while (!content.empty() && (content[0] == '\n' || content[0] == ' '))
          content.erase(0, 1);
        if (answer != nullptr)
          *answer = content;
        const char* finish = r.stopped ? "stop" : "length";
        if (replies != nullptr) {
          std::fprintf(replies,
                       "===== depth %d task %s finish=%s =====\n--- reasoning "
                       "---\n%s\n--- answer ---\n%s\n\n",
                       depth, task, finish,
                       close == std::string::npos
                           ? reply.c_str()
                           : reply.substr(0, close).c_str(),
                       content.c_str());
          std::fflush(replies);
        }
        const double pp = r.prompt_n * 1000.0 / r.prefill_ms;
        const double tg = r.generated * 1000.0 / r.decode_ms;
        const double acc = r.drafted ? 100.0 * r.accepted / r.drafted : 0.0;
        const double wall = (r.prefill_ms + r.decode_ms) / 1000.0;
        std::printf(
            "| %7d | %-8s | %10zu | %7zu | %8.1f | %9.2f | %7d | %-6s | "
            "%7.2f | %8.1f | %7.1f |\n",
            depth, task, r.prompt_n, r.cached, pp, r.prefill_ms / 1000.0,
            r.generated, finish, tg, acc, wall);
        std::fflush(stdout);
        if (csv != nullptr) {
          std::fprintf(
              csv, "a3b,%s,%d,%s,%zu,%zu,%.1f,%.2f,%d,%s,%.2f,%.1f,%.1f\n",
              mode.c_str(), depth, task, r.prompt_n, r.cached, pp,
              r.prefill_ms / 1000.0, r.generated, finish, tg, acc, wall);
          std::fflush(csv);
        }
        table.push_back({depth, task, std::move(r)});
        return true;
      };
      int index = 0;
      for (std::size_t at = 0; at < depths.size(); ++index) {
        const std::size_t comma = depths.find(',', at);
        const int depth = std::atoi(depths.substr(at, comma - at).c_str());
        at = comma == std::string::npos ? depths.size() : comma + 1;
        const std::string summary_user = user_content(depth, 0, index);
        if (!request(depth, task_names[0],
                     encode("<|im_start|>user\n" + summary_user +
                            "<|im_end|>\n" + gen_suffix),
                     kNoCache, nullptr))
          return 1;
        const std::string card_user = user_content(depth, 1, index);
        const std::string card_turn =
            "<|im_start|>user\n" + card_user + "<|im_end|>\n";
        std::string card_answer;
        const bool follow = depth > 0 && depth >= followup_min;
        if (!request(depth, task_names[1], encode(card_turn + gen_suffix),
                     follow ? kSave : kNoCache, &card_answer))
          return 1;
        if (follow &&
            !request(depth, "followup",
                     encode(card_turn + "<|im_start|>assistant\n" +
                            card_answer + "<|im_end|>\n<|im_start|>user\n" +
                            followup + "<|im_end|>\n" + gen_suffix),
                     kResume, nullptr))
          return 1;
      }
      if (replies != nullptr)
        std::fclose(replies);
      if (csv != nullptr)
        std::fclose(csv);
      std::printf("\na3b, Qwen3.6-35B-A3B UD-Q8_K_XL + MTP, %s\n",
                  mode.c_str());
      std::printf("| %-8s | %20s | %9s | %-16s |\n", "task", "test", "t/s",
                  "note");
      std::printf(
          "| :------- | -------------------: | --------: | :--------------- "
          "|\n");
      for (const auto& row : table) {
        const Result& r = row.r;
        char test[64], note[64];
        std::snprintf(test, sizeof(test), "pp%zu @ d%d", r.prompt_n, row.depth);
        if (r.cached > 0)
          std::snprintf(note, sizeof(note), "%zu cached", r.cached);
        else
          note[0] = '\0';
        std::printf("| %-8s | %20s | %9.2f | %-16s |\n", row.task.c_str(), test,
                    r.prompt_n * 1000.0 / r.prefill_ms, note);
        std::snprintf(test, sizeof(test), "tg%d @ d%d", r.generated, row.depth);
        if (r.drafted > 0)
          std::snprintf(note, sizeof(note), "accept %.1f%%",
                        100.0 * r.accepted / r.drafted);
        else
          note[0] = '\0';
        std::printf("| %-8s | %20s | %9.2f | %-16s |\n", row.task.c_str(), test,
                    r.generated * 1000.0 / r.decode_ms, note);
      }
      std::printf("\nmax_tokens %d, thinking %s, follow-up from d%d, ctx %u\n",
                  n_predict, think ? "on" : "off", followup_min,
                  options.max_context);
      std::printf("rows: %s.csv\nreplies: %s.replies.txt\n", stem.c_str(),
                  stem.c_str());
      return 0;
    }
    if (bench == "prefill") {
      // The wide prefill against the 8-row path on corpus slices: speed and
      // the last row's top-64 logits.
      std::FILE* f = std::fopen(corpus_file.c_str(), "rb");
      if (f == nullptr) {
        std::fprintf(stderr, "cannot open %s\n", corpus_file.c_str());
        return 1;
      }
      std::string corpus;
      char buf[65536];
      for (std::size_t got; (got = std::fread(buf, 1, sizeof(buf), f)) > 0;)
        corpus.append(buf, got);
      std::fclose(f);
      const std::uint32_t wide =
          options.prefill_chunk > 0 ? options.prefill_chunk : 2048;
      for (std::size_t at = 0; at < depths.size();) {
        const std::size_t comma = depths.find(',', at);
        const int depth = std::atoi(depths.substr(at, comma - at).c_str());
        at = comma == std::string::npos ? depths.size() : comma + 1;
        const auto ids = render(corpus.substr(
            0, std::min<std::size_t>(corpus.size(),
                                     static_cast<std::size_t>(depth * 3.3))));
        Candidates got[2]{};
        double ms[2]{};
        for (int w = 0; w < 2; ++w) {
          engine->SetPrefillChunk(w == 0 ? 0 : wide);
          engine->Reset();
          const auto p0 = Clock::now();
          if (!engine->SpecPrefill(ids, &got[w], &error)) {
            std::fprintf(stderr, "prefill failed: %s\n", error.c_str());
            return 1;
          }
          ms[w] = Ms(p0, Clock::now());
        }
        double worst = 0;
        int shared = 0;
        for (std::uint32_t i = 0; i < Candidates::kCount; ++i)
          for (std::uint32_t j = 0; j < Candidates::kCount; ++j)
            if (got[0].ids[i] == got[1].ids[j]) {
              ++shared;
              worst = std::max(
                  worst, static_cast<double>(
                             std::fabs(got[0].logits[i] - got[1].logits[j])));
            }
        std::printf(
            "prefill %6zu tok: 8-row %7.1f t/s, wide %7.1f t/s (x%.1f) "
            "| top1 %s, top-64 shared %d, max |dlogit| %.3f (top %.2f)\n",
            ids.size(), ids.size() * 1000.0 / ms[0],
            ids.size() * 1000.0 / ms[1], ms[0] / ms[1],
            got[0].ids[0] == got[1].ids[0] ? "same" : "DIFF", shared, worst,
            got[0].logits[0]);
        std::fflush(stdout);
      }
      return 0;
    }
    if (bench == "rows") {
      // Verify + commit cost per width (graph replay, keep 1 so every
      // width > 1 also pays the rollback), at a few context depths.
      const auto e = tok->Encode(std::string(20000, 'a'));
      std::vector<std::int32_t> filler(e.begin(), e.end());
      std::vector<int> row_depths = {256, 4096, 16384};
      if (depths_set) {
        row_depths.clear();
        for (std::size_t at = 0; at < depths.size();) {
          const std::size_t comma = depths.find(',', at);
          row_depths.push_back(
              std::atoi(depths.substr(at, comma - at).c_str()));
          at = comma == std::string::npos ? depths.size() : comma + 1;
        }
      }
      for (const int depth : row_depths) {
        if (depth + 256 > static_cast<int>(options.max_context))
          break;
        engine->Reset();
        std::vector<std::int32_t> p(prompt_ids);
        while (static_cast<int>(p.size()) < depth)
          p.push_back(filler[p.size() % filler.size()]);
        Candidates last{};
        if (!engine->SpecPrefill(p, &last, &error)) {
          std::fprintf(stderr, "prefill failed: %s\n", error.c_str());
          return 1;
        }
        std::printf("depth %5d:", depth);
        std::vector<Candidates> rows;
        for (std::uint32_t n = 1; n <= models::qwen36_a3b::Engine::kMaxRows;
             ++n) {
          const std::vector<std::int32_t> drafts(n - 1, 13);
          const auto step = [&] {
            return engine->Verify(13, drafts, &rows, &error, mtp > 0) &&
                   engine->SpecCommit(1, &error);
          };
          for (int i = 0; i < 3; ++i)
            if (!step())
              return 1;
          const int reps = 20;
          const auto a0 = Clock::now();
          for (int i = 0; i < reps; ++i)
            if (!step())
              return 1;
          std::printf("  %u:%5.1f", n, Ms(a0, Clock::now()) / reps);
        }
        std::printf("  ms\n");
      }
      return 0;
    }
    if (bench == "lookup") {
      // lookup-probe.py's coding-agent requests: answers that copy from the
      // conversation (file edits, tests of a given module, a YAML edit, a
      // tool plan, a log diagnosis, a follow-up edit of the model's own
      // answer) plus free prose.
      const auto slurp = [](const std::string& path, std::size_t limit) {
        std::string text;
        if (std::FILE* f = std::fopen(path.c_str(), "rb")) {
          char buf[65536];
          for (std::size_t got; (got = std::fread(buf, 1, sizeof(buf), f)) > 0;)
            text.append(buf, got);
          std::fclose(f);
        }
        return text.substr(0, std::min(limit, text.size()));
      };
      const std::string probe = slurp(DataFile("decode-probe.py"), SIZE_MAX);
      const std::string vocab_py =
          slurp(DataFile("make-draft-vocab.py"), SIZE_MAX);
      const std::string ci = slurp(
          DataFile("gufo_experimental/.github/workflows/ci.yml"), SIZE_MAX);
      std::string log =
          slurp(DataFile("bench-real-gufo-sampled-20260925-151423.server.log"),
                SIZE_MAX);
      {
        // Its first 45 lines.
        std::size_t at = 0;
        int lines = 0;
        while (lines < 45 && (at = log.find('\n', at)) != std::string::npos)
          ++at, ++lines;
        if (lines == 45)
          log.resize(at - 1);
      }
      if (probe.empty() || vocab_py.empty() || ci.empty() || log.empty()) {
        std::fprintf(stderr, "lookup bench: missing an input file\n");
        return 1;
      }
      const std::string tools =
          "[\n  {\"name\": \"read_file\", \"parameters\": {\"path\": "
          "\"string\"}},\n"
          "  {\"name\": \"search\", \"parameters\": {\"pattern\": \"string\", "
          "\"glob\": \"string\"}},\n"
          "  {\"name\": \"edit_file\", \"parameters\": {\"path\": \"string\", "
          "\"old\": \"string\", \"new\": \"string\"}},\n"
          "  {\"name\": \"run\", \"parameters\": {\"command\": \"string\", "
          "\"timeout_s\": \"integer\"}}\n]";
      const std::vector<std::pair<const char*, std::vector<std::string>>>
          tasks = {
              {"py-edit",
               {"Here is decode-probe.py:\n\n```python\n" + probe +
                    "\n```\n\nAdd a --tasks-file option that loads the task "
                    "list "
                    "from a JSON file (a list of strings) instead of the "
                    "built-in "
                    "TASKS, keeping everything else unchanged. Return the "
                    "complete "
                    "updated file.",
                "Now also add a --dry-run flag that prints the request bodies "
                "instead of sending them. Return the complete file again."}},
              {"tests",
               {"Here is make-draft-vocab.py:\n\n```python\n" + vocab_py +
                "\n```\n\nWrite pytest unit tests for byte_decoder() and for "
                "the "
                "token filtering rule in main() (refactor the rule into a "
                "function "
                "if needed and show that change too)."}},
              {"yaml-edit",
               {"Here is our CI workflow:\n\n```yaml\n" + ci +
                "\n```\n\nAdd a Windows job that mirrors the existing job but "
                "runs "
                "on windows-latest, caches the build directory, and only runs "
                "on "
                "pushes to main. Return the full updated YAML."}},
              {"tool-plan",
               {"You are a coding agent with these tools:\n" + tools +
                "\n\nTask: in the repository, the function StablePromptPrefix "
                "in "
                "src/cli/serve/inference_backend.cpp mishandles an empty "
                "message "
                "list. Plan the tool calls you would make to find, fix and "
                "test "
                "this. Answer with a JSON array of tool calls only, with "
                "realistic "
                "arguments."}},
              {"log-diag",
               {"Here is the start of a server log:\n\n```\n" + log +
                "\n```\n\nExplain what happened during this load, quote the "
                "lines "
                "that show each phase, and list anything that looks "
                "abnormal."}},
              {"prose",
               {"Explain how speculative decoding with rejection sampling "
                "keeps "
                "the output distribution of the target model exact, for an "
                "engineer who knows probability but not LLMs."}}};
      Result all;
      for (int s = 0; s < seeds; ++s) {
        for (const auto& [name, turns] : tasks) {
          if (!only.empty() &&
              only.find("," + std::string(name) + ",") == std::string::npos)
            continue;
          // Earlier turns carry the reply's content (thinking stripped).
          std::string chat;
          for (std::size_t t = 0; t < turns.size(); ++t) {
            chat += "<|im_start|>user\n" + turns[t] +
                    "<|im_end|>\n<|im_start|>assistant\n";
            std::string text = chat;
            text += think ? "<think>\n" : "<think>\n\n</think>\n\n";
            const auto e = tok->Encode(text);
            const Result r = run(std::vector<std::int32_t>(e.begin(), e.end()),
                                 n_predict, sc.seed + s, false);
            if (!r.ok)
              return 1;
            std::string reply;
            for (const auto id : r.out)
              reply += tok->DecodeTokenCopy(id);
            const std::size_t close = reply.find("</think>");
            std::string content =
                close == std::string::npos ? "" : reply.substr(close + 8);
            while (!content.empty() &&
                   (content[0] == '\n' || content[0] == ' '))
              content.erase(0, 1);
            chat += content + "<|im_end|>\n";
            std::string label = std::string(name);
            if (turns.size() > 1)
              label += "#" + std::to_string(t + 1);
            label += " s" + std::to_string(sc.seed + s);
            std::printf("%5zu prompt  ", r.prompt_n);
            line(label.c_str(), r);
            add(all, r);
          }
        }
      }
      line("ALL", all);
      return 0;
    }
    if (bench == "depth") {
      // longctx-27b.py: a corpus slice (~3.3 chars per token) plus a question.
      std::FILE* f = std::fopen(corpus_file.c_str(), "rb");
      if (f == nullptr) {
        std::fprintf(stderr, "cannot open %s\n", corpus_file.c_str());
        return 1;
      }
      std::string corpus;
      char buf[65536];
      for (std::size_t got; (got = std::fread(buf, 1, sizeof(buf), f)) > 0;)
        corpus.append(buf, got);
      std::fclose(f);
      const std::string question =
          "\n\n---\nAbove is a set of project documents and source files. "
          "List the three most important things a new contributor must know, "
          "citing the files.";
      for (std::size_t at = 0; at < depths.size();) {
        const std::size_t comma = depths.find(',', at);
        const int depth = std::atoi(depths.substr(at, comma - at).c_str());
        at = comma == std::string::npos ? depths.size() : comma + 1;
        const std::string user =
            "[run 0]\n" +
            corpus.substr(
                0, std::min<std::size_t>(
                       corpus.size(), static_cast<std::size_t>(depth * 3.3))) +
            question;
        const auto ids = render(user);
        for (int s = 0; s < seeds; ++s) {
          const Result r = run(ids, n_predict, sc.seed + s, false);
          if (!r.ok)
            return 1;
          std::printf("depth %6zu tok s%llu  prefill %6.1f s %7.1f t/s  |  ",
                      r.prompt_n, static_cast<unsigned long long>(sc.seed + s),
                      r.prefill_ms / 1000.0,
                      r.prompt_n * 1000.0 / r.prefill_ms);
          line("decode", r);
        }
      }
      return 0;
    }
    const Result r = run(prompt_ids, n_predict, sc.seed, true);
    if (!r.ok)
      return 1;
    engine->PrintSpecProfile();
    std::fprintf(stderr,
                 "\n\na3b: %d tokens in %.0f ms = %.2f t/s; %zu steps "
                 "(%.2f tokens/step, %.2f ms/step), drafts %zu accepted %zu "
                 "(%.1f%%)\n",
                 r.generated, r.decode_ms, r.generated * 1000.0 / r.decode_ms,
                 r.cycles,
                 r.cycles ? static_cast<double>(r.generated) / r.cycles : 0.0,
                 r.cycles ? r.decode_ms / r.cycles : 0.0, r.drafted, r.accepted,
                 r.drafted ? 100.0 * r.accepted / r.drafted : 0.0);
    return 0;
  }

  const std::uint32_t vocab = engine->config().vocab;
  std::vector<float> logits(vocab);
  const auto p0 = Clock::now();
  for (std::size_t off = 0; off < prompt_ids.size(); off += chunk) {
    const std::size_t n = std::min<std::size_t>(chunk, prompt_ids.size() - off);
    const bool last = off + n == prompt_ids.size();
    if (!engine->Forward({prompt_ids.data() + off, n},
                         last ? logits.data() : nullptr, &error)) {
      std::fprintf(stderr, "prefill failed: %s\n", error.c_str());
      return 1;
    }
  }
  const auto p1 = Clock::now();
  const double prefill_ms = Ms(p0, p1);
  std::fprintf(stderr, "a3b: prompt %zu tokens in %.0f ms (%.1f t/s)\n\n",
               prompt_ids.size(), prefill_ms,
               prompt_ids.size() * 1000.0 / prefill_ms);
  if (ablate > 0) {
    // Graph-replayed decode with one phase removed at a time; the delta to
    // the full step is what that phase costs in the real schedule.
    static constexpr const char* kNames[] = {
        "embed",       "norm+add",  "gdn proj",     "gdn core", "gdn out",
        "attn proj",   "attn core", "attn out",     "router",   "shared exp",
        "exp gate/up", "exp down",  "moe epilogue", "head"};
    const auto run = [&](std::uint32_t mask) {
      engine->SetSkip(mask);
      const std::int32_t t = 13;
      for (int i = 0; i < 3; ++i)
        (void)engine->Forward({&t, 1}, logits.data(), &error);
      const auto a0 = Clock::now();
      for (int i = 0; i < ablate; ++i)
        (void)engine->Forward({&t, 1}, logits.data(), &error);
      return Ms(a0, Clock::now()) / ablate;
    };
    // Masks: full, then per phase "without it" and "only it", then nothing.
    // Rounds alternate so clock drift hits every mask alike; medians.
    constexpr int kP = models::qwen36_a3b::kPhCount;
    constexpr std::uint32_t kAll = (1u << kP) - 2;  // embed always runs
    std::vector<std::uint32_t> masks{0};
    for (int p = 1; p < kP; ++p)
      masks.push_back(1u << p);
    for (int p = 1; p < kP; ++p)
      masks.push_back(kAll & ~(1u << p));
    masks.push_back(kAll);
    constexpr int kRounds = 3;
    std::vector<std::vector<double>> ms(masks.size());
    for (int r = 0; r < kRounds; ++r)
      for (std::size_t m = 0; m < masks.size(); ++m)
        ms[m].push_back(run(masks[m]));
    const auto med = [&](std::size_t m) {
      auto v = ms[m];
      std::sort(v.begin(), v.end());
      return v[v.size() / 2];
    };
    const double full = med(0);
    const double none = med(masks.size() - 1);
    std::fprintf(stderr,
                 "ablation, %d tokens x %d rounds (medians): full %.3f ms, "
                 "empty %.3f ms\n  %-13s %9s %9s\n",
                 ablate, kRounds, full, none, "phase", "without", "alone");
    double sum_without = 0, sum_alone = 0;
    for (int p = 1; p < kP; ++p) {
      const double without = full - med(p);
      const double alone = med(kP - 1 + p) - none;
      sum_without += without;
      sum_alone += alone;
      std::fprintf(stderr, "  %-13s %9.3f %9.3f\n", kNames[p], without, alone);
    }
    std::fprintf(stderr, "  %-13s %9.3f %9.3f\n", "sum", sum_without,
                 sum_alone);
    return 0;
  }
  std::fprintf(stdout, "%s", raw ? "" : "<think>\n");

  sampling::SamplerState sampler(sc);
  sampler.ResetHistory({ids.data(), ids.size()});
  double gpu_ms = 0;
  double sample_ms = 0;
  int generated = 0;
  const auto d0 = Clock::now();
  for (; generated < n_predict; ++generated) {
    const auto s0 = Clock::now();
    const auto next = sampler.Sample(logits);
    sampler.Accept(next);
    sample_ms += Ms(s0, Clock::now());
    if (next == eos || (im_end && next == *im_end))
      break;
    const std::string piece = tok->DecodeTokenCopy(next);
    std::fwrite(piece.data(), 1, piece.size(), stdout);
    std::fflush(stdout);
    const auto g0 = Clock::now();
    const std::int32_t t = static_cast<std::int32_t>(next);
    if (!engine->Forward({&t, 1}, logits.data(), &error)) {
      std::fprintf(stderr, "\ndecode failed: %s\n", error.c_str());
      return 1;
    }
    gpu_ms += Ms(g0, Clock::now());
  }
  const double total_ms = Ms(d0, Clock::now());
  std::fprintf(stderr,
               "\n\na3b: %d tokens in %.0f ms = %.2f t/s "
               "(forward %.2f ms/token = %.1f t/s, sampling %.2f ms/token)\n",
               generated, total_ms, generated * 1000.0 / total_ms,
               generated ? gpu_ms / generated : 0.0,
               gpu_ms > 0 ? generated * 1000.0 / gpu_ms : 0.0,
               generated ? sample_ms / generated : 0.0);
  engine->PrintProfile();
  return 0;
}
