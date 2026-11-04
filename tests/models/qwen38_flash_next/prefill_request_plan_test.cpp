// The expert cut covers a request's new rows only when there are many of them.
// A warm agent turn re-reads a few hundred new tokens: cutting them gains less
// than the 256-row full-precision tail costs the next turn, so such a request
// must run exactly as the normal mode does.
#include <algorithm>
#include <cstddef>
#include <iostream>
#include <stdexcept>

#include "src/models/qwen38_flash_next/prefill_tail.hpp"

namespace q = gufo::models::qwen38_flash_next;

namespace {
void Require(bool condition, const char* message) {
  if (!condition)
    throw std::runtime_error(message);
}

constexpr std::size_t kTail = 256;
constexpr std::size_t kMin = q::kPrefillExpertMinNewTokens;

// Rows of one request the cut policy computes, the way Session::Feed walks
// them: passes of `chunk` rows, each cut while it starts below the boundary.
std::size_t CutRows(std::size_t prompt, std::size_t reused, std::size_t tail,
                    std::size_t chunk) {
  const auto boundary = q::PrefillExpertRequestBoundary(prompt, reused, tail);
  std::size_t cut = 0;
  for (std::size_t offset = reused; offset < prompt;) {
    const auto n = q::ClipPrefillExpertTail(offset, std::min(chunk, prompt - offset),
                                            boundary);
    if (q::ApplyPrefillExpertPolicy(offset, boundary, tail))
      cut += n;
    offset += n;
  }
  return cut;
}
}  // namespace

int main() {
  try {
    Require(kMin > kTail,
            "the threshold must exceed the tail, or a cut request has no cut rows");
    // Without a tail every row is cut, whatever the request length.
    Require(q::PrefillExpertCutsRequest(100, 0, 0) &&
                q::PrefillExpertCutsRequest(100000, 99990, 0),
            "a configuration without a tail cuts every request");
    Require(q::PrefillExpertRequestBoundary(5000, 0, 0) == 5000,
            "no tail: boundary is the prompt end, as before");

    // The threshold counts new rows, not the prompt.
    Require(!q::PrefillExpertCutsRequest(kMin - 1, 0, kTail),
            "one row below the threshold must not cut");
    Require(q::PrefillExpertCutsRequest(kMin, 0, kTail),
            "exactly the threshold cuts");
    Require(!q::PrefillExpertCutsRequest(60000, 60000 - kMin + 1, kTail),
            "a long prompt with a short new part must not cut");
    Require(q::PrefillExpertCutsRequest(60000, 60000 - kMin, kTail),
            "a long prompt with a long new part cuts");
    Require(!q::PrefillExpertCutsRequest(1000, 1000, kTail) &&
                !q::PrefillExpertCutsRequest(1000, 5000, kTail),
            "a fully reused prompt has no new rows (and a restore past the prompt "
            "does not underflow)");

    // A request that does not cut runs every pass under the normal arithmetic:
    // boundary zero applies the policy to no offset and clips no pass.
    for (const std::size_t prompt : {2048u, 5534u, 9000u, 38185u}) {
      const std::size_t reused = prompt > 800 ? prompt - 750 : 0;
      Require(q::PrefillExpertRequestBoundary(prompt, reused, kTail) == 0,
              "a warm turn has no policy boundary");
      Require(CutRows(prompt, reused, kTail, 4096) == 0,
              "a warm turn computed cut rows");
    }
    Require(CutRows(kMin - 1, 0, kTail, 4096) == 0,
            "a cold prompt below the threshold computed cut rows");

    // A cold long prompt cuts everything except the last `tail` rows, exactly as
    // before.
    for (const std::size_t prompt :
         {kMin, std::size_t{22167}, std::size_t{32797}, std::size_t{100000}}) {
      Require(q::PrefillExpertRequestBoundary(prompt, 0, kTail) == prompt - kTail,
              "a cold long prompt keeps its last 256 rows in full precision");
      Require(CutRows(prompt, 0, kTail, 4096) == prompt - kTail,
              "a cold long prompt must cut all but the tail");
    }
    // A long request that restored a prefix cuts its new rows only.
    Require(CutRows(40000, 12288, kTail, 4096) == 40000 - kTail - 12288,
            "a restored long request cuts its new rows except the tail");

    // The warm ten-turn agent session of the rival run: a 5.5K first prompt is a
    // long read and cuts once; the turns after it add about 750 new tokens each,
    // never reach the threshold, and compute the same rows, with the same
    // arithmetic, as the normal mode.
    std::size_t prompt = 5534, reused = 0;
    Require(q::PrefillExpertCutsRequest(prompt, reused, kTail),
            "the first 5.5K prompt of the session reads enough new rows to cut");
    reused = prompt;
    prompt += 750;
    for (int turn = 1; turn < 10; ++turn) {
      Require(!q::PrefillExpertCutsRequest(prompt, reused, kTail),
              "a warm turn must never cut");
      reused = prompt;
      prompt += 750;
    }
    // A session whose first prompt is shorter than the threshold never cuts.
    prompt = 2923;
    reused = 0;
    for (int turn = 0; turn < 10; ++turn) {
      Require(!q::PrefillExpertCutsRequest(prompt, reused, kTail),
              "a session that starts below the threshold must never cut");
      reused = prompt;
      prompt += 750;
    }
    // A long first paste cuts once; the questions after it do not.
    Require(q::PrefillExpertCutsRequest(40000, 0, kTail),
            "a pasted 40K document cuts");
    Require(!q::PrefillExpertCutsRequest(40800, 40000, kTail),
            "the follow-up question after it does not");

    std::cout << "prefill request plan tests passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
