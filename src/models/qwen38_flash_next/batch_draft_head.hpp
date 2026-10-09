#ifndef GUFO_MODELS_QWEN38_FLASH_NEXT_BATCH_DRAFT_HEAD_HPP_
#define GUFO_MODELS_QWEN38_FLASH_NEXT_BATCH_DRAFT_HEAD_HPP_

#include <cstdint>

namespace gufo::models::qwen38_flash_next {

inline bool BatchDraftHeadSwitch(const char* value) {
  return value != nullptr && value[0] == '1' && value[1] == '\0';
}

enum class BatchDraftHead { kPlain, kDraft, kSubset, kSubsetDraft };

// One item retains the single-session head regardless of the batch switch.
inline BatchDraftHead SelectBatchDraftHead(bool enabled, bool subset,
                                           bool draft, std::uint32_t items) {
  if (!enabled && items >= 2)
    return BatchDraftHead::kPlain;
  if (subset)
    return draft ? BatchDraftHead::kSubsetDraft : BatchDraftHead::kSubset;
  return draft ? BatchDraftHead::kDraft : BatchDraftHead::kPlain;
}

// A cohort's head: the subset only when every session in it drafts in latin
// mode (`any_full`: some session is in DraftVocabulary::kAuto full mode).
// One item then matches Executor::MtpHead (subset = has_subset && !full).
inline BatchDraftHead SelectCohortDraftHead(bool enabled, bool has_subset,
                                            bool any_full, bool draft,
                                            std::uint32_t items) {
  return SelectBatchDraftHead(enabled, has_subset && !any_full, draft, items);
}

inline bool BatchDraftSubset(BatchDraftHead head) {
  return head == BatchDraftHead::kSubset ||
         head == BatchDraftHead::kSubsetDraft;
}

inline const char* BatchDraftHeadName(BatchDraftHead head) {
  switch (head) {
    case BatchDraftHead::kSubsetDraft:
      return "subset+draft_out";
    case BatchDraftHead::kSubset:
      return "subset";
    case BatchDraftHead::kDraft:
      return "draft_out";
    default:
      return "plain";
  }
}

inline std::uint32_t DraftRemapCount(BatchDraftHead head, std::uint32_t rows,
                                     std::uint32_t candidates, bool argmax) {
  if (!BatchDraftSubset(head))
    return 0;
  return argmax ? 1u : (rows < candidates ? rows : candidates);
}

}  // namespace gufo::models::qwen38_flash_next
#endif
