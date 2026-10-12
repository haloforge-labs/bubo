#include "src/models/qwen38_flash_next/batch_draft_head.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>

namespace q = gufo::models::qwen38_flash_next;

void Require(bool ok) {
  if (!ok)
    throw std::runtime_error("batch draft head contract failed");
}

int main() {
  Require(!q::BatchDraftHeadSwitch(nullptr));
  for (const char* value : {"", "0", "true", "01", "11", "1 ", " 1"})
    Require(!q::BatchDraftHeadSwitch(value));
  Require(q::BatchDraftHeadSwitch("1"));
  unsigned combinations = 0;
  for (bool enabled : {false, true})
    for (bool subset : {false, true})
      for (bool draft : {false, true})
        for (unsigned items : {1u, 2u, 4u, 8u}) {
          const auto head =
              q::SelectBatchDraftHead(enabled, subset, draft, items);
          const auto expected =
              !enabled && items >= 2 ? q::BatchDraftHead::kPlain
              : subset && draft      ? q::BatchDraftHead::kSubsetDraft
              : subset               ? q::BatchDraftHead::kSubset
              : draft                ? q::BatchDraftHead::kDraft
                                     : q::BatchDraftHead::kPlain;
          Require(head == expected);
          const unsigned rows = q::BatchDraftSubset(head) ? 130202 : 248320;
          Require(rows ==
                  (subset && (enabled || items == 1) ? 130202u : 248320u));
          for (unsigned vocab : {3u, 64u, 130202u, 248320u}) {
            Require(
                q::DraftRemapCount(head, vocab, 64, false) ==
                (q::BatchDraftSubset(expected) ? std::min(vocab, 64u) : 0u));
            Require(q::DraftRemapCount(head, vocab, 64, true) ==
                    (q::BatchDraftSubset(expected) ? 1u : 0u));
          }
          ++combinations;
        }
  // Cohort heads (earlier candidate build): a singleton picks what Executor::MtpHead picks
  // (subset = has_subset && !full; else draft_out if loaded, else output),
  // with or without the batch switch; a cohort with any full-mode session
  // never uses the subset; a plain cohort of two or more keeps kPlain.
  unsigned cohort = 0;
  for (bool enabled : {false, true})
    for (bool has_subset : {false, true})
      for (bool any_full : {false, true})
        for (bool draft : {false, true})
          for (unsigned items : {1u, 2u, 4u}) {
            const auto head = q::SelectCohortDraftHead(enabled, has_subset,
                                                       any_full, draft, items);
            if (items == 1) {
              const bool subset = has_subset && !any_full;
              Require(head == (subset   ? (draft ? q::BatchDraftHead::kSubsetDraft
                                                 : q::BatchDraftHead::kSubset)
                               : draft ? q::BatchDraftHead::kDraft
                                       : q::BatchDraftHead::kPlain));
            }
            if (any_full)
              Require(!q::BatchDraftSubset(head));
            if (!enabled && items >= 2)
              Require(head == q::BatchDraftHead::kPlain);
            ++cohort;
          }
  // CPU replay of the existing RemapIdsKernel formula; GPU execution is G2.
  const std::int32_t map[] = {17, 502, 32000};
  for (auto head :
       {q::BatchDraftHead::kPlain, q::BatchDraftHead::kDraft,
        q::BatchDraftHead::kSubset, q::BatchDraftHead::kSubsetDraft}) {
    std::array<std::uint32_t, 4> ids{2, 0, 1, 0xdeadbeef};
    const auto original = ids;
    for (unsigned i = 0; i < q::DraftRemapCount(head, 3, 64, false); ++i)
      ids[i] = static_cast<std::uint32_t>(map[ids[i]]);
    if (q::BatchDraftSubset(head))
      Require(ids[0] == 32000 && ids[1] == 17 && ids[2] == 502);
    else
      Require(ids == original);
    Require(ids[3] == 0xdeadbeef);
    // token+candidates consumes this already remapped first candidate.
    Require(ids[0] == (q::BatchDraftSubset(head) ? 32000u : 2u));
    std::array<std::uint32_t, 2> argmax{1, 0xdeadbeef};
    for (unsigned i = 0; i < q::DraftRemapCount(head, 3, 64, true); ++i)
      argmax[i] = static_cast<std::uint32_t>(map[argmax[i]]);
    Require(argmax[0] == (q::BatchDraftSubset(head) ? 502u : 1u));
    Require(argmax[1] == 0xdeadbeef);
  }
  std::cout
      << "PASS " << combinations
      << " switch x latin x draft_out x items combinations; " << cohort
      << " cohort-head (auto full mode) combinations; strict parser; "
         "candidate/argmax remap, sentinels and OFF identity\n";
}
