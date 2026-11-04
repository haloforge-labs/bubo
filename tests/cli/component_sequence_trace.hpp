#ifndef GUFO_TESTS_CLI_COMPONENT_SEQUENCE_TRACE_HPP_
#define GUFO_TESTS_CLI_COMPONENT_SEQUENCE_TRACE_HPP_

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>

// Recorded operator evidence, not model token IDs or invented successful
// behavior. Requests follow the exact points-deep / points-fork manifest order.
// A CPU fake may assign abstract values to each lineage and preserve the
// recorded shared prefix; it must not claim those values are real tokenizer IDs.
namespace gufo::tests::component_sequence_trace {

inline constexpr std::uint32_t kUnknownTokens =
    std::numeric_limits<std::uint32_t>::max();
inline constexpr std::uint64_t kCacheCapacity = 4ULL << 30;
inline constexpr std::uint64_t kPayloadBlockBytes = 65536;
inline constexpr std::uint64_t kBlockMetadataBytes = 1024;
inline constexpr std::uint64_t kPoolHeaderBytes = 4096;
inline constexpr std::uint32_t kAppendChunkRows = 256;
inline constexpr std::uint32_t kPrefillChunkRows = 4096;
inline constexpr std::uint32_t kFullTailRows = 256;
inline constexpr std::uint64_t kBackingBudget =
    (kCacheCapacity - (4ULL << 20)) / 10 * 9;
inline constexpr std::uint64_t kBackingBlocks =
    (kBackingBudget - kPoolHeaderBytes) /
    (kPayloadBlockBytes + kBlockMetadataBytes);
inline constexpr std::uint64_t kPayloadCapacity =
    kBackingBlocks * kPayloadBlockBytes;
inline constexpr std::uint64_t kBackingMetadataBytes =
    kPoolHeaderBytes + kBackingBlocks * kBlockMetadataBytes;
static_assert(kBackingBlocks == 58018);
static_assert(kPayloadCapacity == 3802267648ULL);
static_assert(kBackingMetadataBytes == 59414528ULL);

// ProductionGeometry() in continuation_layout_test.cpp, independently checked
// against ContinuationLayout: 48 trunk layers; 36 linear, 12 attention; hc_dim
// 4*2560; KV dim 2*256; indexer dim 128, ratio 4, raw ring 2048; vocabulary
// 248320; PLE conv history (4-1)*3; seven drafts + one anchor = eight kept rows.
// Logical bytes differ from rounded committed-payload bytes. In particular the
// small token component still consumes a 64 KiB block for each 256-row chunk.
enum class Part : std::uint8_t {
  kTokens, kLogits, kMetadata, kConvolution, kRecurrent, kKey, kValue,
  kPooledKeys, kRawKeys, kPleHistory, kDraftKey, kDraftValue,
  kDraftPooledKeys, kDraftRawKeys, kDraftResidual, kKeptHidden,
};
struct ComponentGeometry {
  Part part;
  std::uint32_t layer;
  std::uint32_t row_bytes;
  std::uint32_t rows_per_chunk;
  std::uint32_t state_bytes;
  bool draft;
  bool pooled;
};
inline constexpr auto kComponents = [] {
  std::array<ComponentGeometry, 130> components{};
  std::size_t n = 0;
  const auto add = [&](Part part, std::uint32_t row_bytes,
                       std::uint32_t state_bytes, std::uint32_t layer = 0,
                       bool draft = false, bool pooled = false) {
    components[n++] = {part, layer, row_bytes,
                       row_bytes ? kAppendChunkRows : 0, state_bytes,
                       draft, pooled};
  };
  add(Part::kTokens, 4, 0);
  add(Part::kLogits, 0, 248320 * 4);
  add(Part::kMetadata, 0, 160);
  for (std::uint32_t layer = 0; layer < 48; ++layer) {
    if ((layer + 1) % 4 != 0) {
      add(Part::kConvolution, 0, 3 * 10240 * 4, layer);
      add(Part::kRecurrent, 0, 48 * 128 * 128 * 4, layer);
    } else {
      add(Part::kKey, 1024, 0, layer);
      add(Part::kValue, 1024, 0, layer);
      add(Part::kPooledKeys, 256, 0, layer, false, true);
      add(Part::kRawKeys, 0, 2048 * 128 * 4, layer);
    }
  }
  add(Part::kPleHistory, 0, 9 * 10240 * 4);
  add(Part::kDraftKey, 1024, 0, 0, true);
  add(Part::kDraftValue, 1024, 0, 0, true);
  add(Part::kDraftPooledKeys, 256, 0, 0, true, true);
  add(Part::kDraftRawKeys, 0, 2048 * 128 * 4, 0, true);
  add(Part::kDraftResidual, 0, 10240 * 4, 0, true);
  add(Part::kKeptHidden, 0, 8 * 10240 * 4);
  return components;
}();
inline constexpr std::uint64_t kPrivateLogicalBytes = [] {
  std::uint64_t bytes = 0;
  for (const auto& c : kComponents)
    bytes += c.state_bytes;
  return bytes;
}();
inline constexpr std::uint64_t kPrivateRoundedBytes = [] {
  std::uint64_t bytes = 0;
  for (const auto& c : kComponents)
    if (c.state_bytes)
      bytes += (c.state_bytes + kPayloadBlockBytes - 1) /
               kPayloadBlockBytes * kPayloadBlockBytes;
  return bytes;
}();
static_assert(kPrivateLogicalBytes == 133032096ULL);
static_assert(kPrivateRoundedBytes == 133496832ULL);
constexpr std::uint64_t LogicalSnapshotBytes(std::uint32_t target,
                                            std::uint32_t draft,
                                            std::uint32_t blocks,
                                            std::uint32_t draft_blocks) {
  return kPrivateLogicalBytes + std::uint64_t{target} * 24580 +
         std::uint64_t{blocks} * 3072 + std::uint64_t{draft} * 2048 +
         std::uint64_t{draft_blocks} * 256;
}
// Exact operator B checkpoint from the FL3T256 failure report.
static_assert(LogicalSnapshotBytes(18068, 18067, 4517, 4516) == 629177072ULL);

enum class Reuse : std::uint8_t { kNone, kRam, kLive, kUnknown };
enum class EventKind : std::uint8_t {
  kCapture, kEvict, kRefused, kSkipped,
};
struct Event {
  EventKind kind;
  std::uint32_t tokens;
  std::uint8_t rank;  // 255 when the log does not emit a rank.
  std::string_view reason;  // Original numeric/string reason, unchanged.
  // Evict: unique_bytes_freed. Skipped: snapshot_skipped.bytes. Capture and
  // Refused do not emit payload bytes in b0; zero means absent, not measured 0.
  std::uint64_t bytes;
  std::uint64_t ledger_bytes;  // Zero when absent from that event.
};
struct Request {
  std::string_view point_id;
  std::uint8_t lineage;
  std::int8_t parent_lineage;  // -1 for a nonfork session.
  std::uint32_t length;
  std::uint32_t shared_prefix_tokens;
  // Observed b0 P is a minimum regression oracle, not the optimal feasible P.
  // In particular deep row204 p05 is observed cold; a new admission test must
  // additionally require P>0 when an earlier feasible checkpoint exists.
  std::uint32_t minimum_reuse;
  std::uint32_t observed_prefill;
  std::uint32_t generated_tokens;
  std::uint64_t restored_bytes;
  Reuse reuse;
  std::uint32_t event_begin;
  std::uint32_t event_count;
  std::uint16_t status;  // 0 = connection reset; completion never arrived.
};
inline constexpr std::array<std::string_view, 16> kLineages = {
    "more-ernie-row1086",  // 0
    "more-ernie-row204",  // 1
    "more-ernie-row304",  // 2
    "more-ernie-row3427",  // 3
    "more-ernie-row343",  // 4
    "more-ernie-row393",  // 5
    "more-ernie-row872",  // 6
    "more-nvidia-33102",  // 7
    "more-ernie-row3427--fork-p02",  // 8
    "more-ernie-row3427--fork-p04",  // 9
    "more-ernie-row393--fork-p02",  // 10
    "more-ernie-row393--fork-p04",  // 11
    "more-ernie-row304--fork-p02",  // 12
    "more-ernie-row304--fork-p04",  // 13
    "more-nvidia-33102--fork-p02",  // 14
    "more-nvidia-33102--fork-p04",  // 15
};

// server.err of recorded run pf-merge-deep-4g-c0, arm C0
// SHA256 4bf4941859fa570457907343b428d0a5d9ef9ddb1c32ec992bd24f4b33acfafb
inline constexpr std::array<Event, 308> kDeepEvents = {{
    {EventKind::kCapture, 9059, 3, "0", 0ULL, 2289628180ULL},
    {EventKind::kCapture, 20480, 3, "0", 0ULL, 2684171260ULL},
    {EventKind::kCapture, 43008, 3, "0", 0ULL, 3079775164ULL},
    {EventKind::kCapture, 87339, 0, "0", 0ULL, 3870173960ULL},
    {EventKind::kCapture, 96256, 3, "0", 0ULL, 3871530232ULL},
    {EventKind::kCapture, 97453, 3, "0", 0ULL, 3872225060ULL},
    {EventKind::kCapture, 97460, 0, "0", 0ULL, 3872829916ULL},
    {EventKind::kEvict, 87339, 0, "1", 545388ULL, 3874859432ULL},
    {EventKind::kEvict, 97453, 3, "1", 604772ULL, 3874254660ULL},
    {EventKind::kCapture, 106496, 3, "0", 0ULL, 3873114764ULL},
    {EventKind::kEvict, 96256, 1, "1", 598000ULL, 3874770820ULL},
    {EventKind::kCapture, 107024, 3, "0", 0ULL, 3873219252ULL},
    {EventKind::kEvict, 43008, 1, "1", 284336ULL, 3874563708ULL},
    {EventKind::kCapture, 107031, 0, "0", 0ULL, 3873596344ULL},
    {EventKind::kEvict, 97460, 0, "1", 604800ULL, 3874620640ULL},
    {EventKind::kCapture, 107069, 3, "0", 0ULL, 3873653372ULL},
    {EventKind::kEvict, 107031, 0, "1", 661372ULL, 3875796640ULL},
    {EventKind::kEvict, 107024, 2, "1", 661344ULL, 3875135296ULL},
    {EventKind::kEvict, 106496, 1, "1", 658320ULL, 3874476976ULL},
    {EventKind::kCapture, 116736, 3, "0", 0ULL, 3873216272ULL},
    {EventKind::kEvict, 107069, 2, "1", 661524ULL, 3875016940ULL},
    {EventKind::kCapture, 117311, 3, "0", 0ULL, 3873318120ULL},
    {EventKind::kEvict, 20480, 1, "1", 151632ULL, 3874945336ULL},
    {EventKind::kCapture, 117318, 0, "0", 0ULL, 3873888424ULL},
    {EventKind::kEvict, 117311, 3, "1", 721852ULL, 3876058044ULL},
    {EventKind::kEvict, 116736, 3, "1", 718640ULL, 3875339404ULL},
    {EventKind::kEvict, 9059, 3, "1", 84060ULL, 3875255344ULL},
    {EventKind::kCapture, 124928, 3, "0", 0ULL, 3873782632ULL},
    {EventKind::kEvict, 117318, 0, "1", 721880ULL, 3875751056ULL},
    {EventKind::kCapture, 126707, 3, "0", 0ULL, 3873967252ULL},
    {EventKind::kRefused, 126714, 255, "byte_capacity", 0ULL, 0ULL},
    {EventKind::kSkipped, 126714, 255, "3", 140319368ULL, 0ULL},
    {EventKind::kEvict, 124928, 1, "1", 766896ULL, 3876339276ULL},
    {EventKind::kRefused, 135168, 255, "byte_capacity", 0ULL, 0ULL},
    {EventKind::kSkipped, 135168, 255, "3", 372467616ULL, 0ULL},
    {EventKind::kEvict, 126707, 2, "1", 1283688ULL, 3875137868ULL},
    {EventKind::kRefused, 136485, 255, "byte_capacity", 0ULL, 0ULL},
    {EventKind::kSkipped, 136485, 255, "3", 408631860ULL, 0ULL},
    {EventKind::kRefused, 136579, 255, "byte_capacity", 0ULL, 0ULL},
    {EventKind::kSkipped, 136579, 255, "3", 3883486892ULL, 0ULL},
    {EventKind::kCapture, 4096, 3, "0", 0ULL, 3862120736ULL},
    {EventKind::kCapture, 8192, 3, "0", 0ULL, 3862546704ULL},
    {EventKind::kCapture, 17429, 3, "0", 0ULL, 3863462652ULL},
    {EventKind::kCapture, 17436, 0, "0", 0ULL, 3863596404ULL},
    {EventKind::kCapture, 22528, 3, "0", 0ULL, 3864191284ULL},
    {EventKind::kCapture, 32768, 3, "0", 0ULL, 3865282100ULL},
    {EventKind::kCapture, 44936, 3, "0", 0ULL, 3866592524ULL},
    {EventKind::kCapture, 45709, 0, "0", 0ULL, 3866959656ULL},
    {EventKind::kCapture, 46879, 3, "0", 0ULL, 3867371940ULL},
    {EventKind::kCapture, 46920, 0, "0", 0ULL, 3867679404ULL},
    {EventKind::kCapture, 49620, 3, "0", 0ULL, 3868224420ULL},
    {EventKind::kCapture, 71680, 3, "0", 0ULL, 3870561412ULL},
    {EventKind::kEvict, 17436, 0, "1", 133696ULL, 3873995804ULL},
    {EventKind::kEvict, 45709, 0, "1", 299924ULL, 3873695880ULL},
    {EventKind::kEvict, 49620, 3, "1", 322880ULL, 3873373000ULL},
    {EventKind::kEvict, 46879, 3, "1", 306972ULL, 3873066028ULL},
    {EventKind::kEvict, 44936, 3, "1", 295328ULL, 3872770700ULL},
    {EventKind::kEvict, 32768, 3, "1", 224016ULL, 3872546684ULL},
    {EventKind::kEvict, 22528, 3, "1", 163696ULL, 3872382988ULL},
    {EventKind::kCapture, 106496, 3, "0", 0ULL, 3872420340ULL},
    {EventKind::kEvict, 8192, 1, "1", 79248ULL, 3876758596ULL},
    {EventKind::kEvict, 4096, 1, "1", 55120ULL, 3876703476ULL},
    {EventKind::kRefused, 143689, 255, "byte_capacity", 0ULL, 0ULL},
    {EventKind::kSkipped, 143689, 255, "3", 1154938820ULL, 0ULL},
    {EventKind::kEvict, 46920, 0, "1", 307136ULL, 3876374236ULL},
    {EventKind::kEvict, 71680, 3, "1", 453232ULL, 3875921004ULL},
    {EventKind::kEvict, 17429, 3, "1", 133668ULL, 3875787336ULL},
    {EventKind::kRefused, 143360, 255, "byte_capacity", 0ULL, 0ULL},
    {EventKind::kSkipped, 143360, 255, "3", 1145905056ULL, 0ULL},
    {EventKind::kEvict, 106496, 1, "1", 1084304ULL, 3874752184ULL},
    {EventKind::kRefused, 144154, 255, "byte_capacity", 0ULL, 0ULL},
    {EventKind::kSkipped, 144154, 255, "3", 1167706888ULL, 0ULL},
    {EventKind::kRefused, 144214, 255, "byte_capacity", 0ULL, 0ULL},
    {EventKind::kSkipped, 144214, 255, "3", 4093144824ULL, 0ULL},
    {EventKind::kCapture, 36864, 3, "0", 0ULL, 3865087520ULL},
    {EventKind::kCapture, 75776, 3, "0", 0ULL, 3868858720ULL},
    {EventKind::kRefused, 153698, 255, "byte_capacity", 0ULL, 0ULL},
    {EventKind::kSkipped, 153698, 255, "3", 2273356328ULL, 0ULL},
    {EventKind::kRefused, 153742, 255, "byte_capacity", 0ULL, 0ULL},
    {EventKind::kSkipped, 153742, 255, "3", 2274564568ULL, 0ULL},
    {EventKind::kCapture, 4096, 3, "0", 0ULL, 3868525760ULL},
    {EventKind::kCapture, 8192, 3, "0", 0ULL, 3868951728ULL},
    {EventKind::kCapture, 15862, 3, "0", 0ULL, 3869705920ULL},
    {EventKind::kCapture, 18302, 0, "0", 0ULL, 3870055800ULL},
    {EventKind::kCapture, 26624, 3, "0", 0ULL, 3870962208ULL},
    {EventKind::kEvict, 75776, 3, "1", 1331904ULL, 3870833112ULL},
    {EventKind::kCapture, 36864, 3, "0", 0ULL, 3870745248ULL},
    {EventKind::kCapture, 37731, 3, "0", 0ULL, 3871059524ULL},
    {EventKind::kCapture, 37738, 0, "0", 0ULL, 3871312564ULL},
    {EventKind::kCapture, 40960, 3, "0", 0ULL, 3869807236ULL},
    {EventKind::kEvict, 18302, 0, "1", 138456ULL, 3871404308ULL},
    {EventKind::kEvict, 36864, 3, "1", 1057200ULL, 3870347108ULL},
    {EventKind::kCapture, 55296, 3, "0", 0ULL, 3870181820ULL},
    {EventKind::kCapture, 56128, 3, "0", 0ULL, 3870604252ULL},
    {EventKind::kCapture, 56475, 0, "0", 0ULL, 3870996544ULL},
    {EventKind::kCapture, 73728, 3, "0", 0ULL, 3870979192ULL},
    {EventKind::kEvict, 37738, 0, "1", 252984ULL, 3872376648ULL},
    {EventKind::kCapture, 75374, 3, "0", 0ULL, 3871329512ULL},
    {EventKind::kEvict, 56128, 2, "1", 361392ULL, 3872133656ULL},
    {EventKind::kCapture, 75381, 0, "0", 0ULL, 3871442932ULL},
    {EventKind::kEvict, 56475, 0, "1", 363420ULL, 3873792696ULL},
    {EventKind::kEvict, 75374, 2, "1", 474728ULL, 3873317968ULL},
    {EventKind::kEvict, 73728, 1, "1", 465296ULL, 3872852672ULL},
    {EventKind::kEvict, 55296, 1, "1", 356720ULL, 3872495952ULL},
    {EventKind::kEvict, 40960, 1, "1", 272272ULL, 3872223680ULL},
    {EventKind::kCapture, 94285, 3, "0", 0ULL, 3871704364ULL},
    {EventKind::kEvict, 37731, 2, "1", 252956ULL, 3872893952ULL},
    {EventKind::kCapture, 94292, 0, "0", 0ULL, 3872037784ULL},
    {EventKind::kEvict, 75381, 0, "1", 474756ULL, 3874619780ULL},
    {EventKind::kEvict, 94285, 3, "1", 586292ULL, 3874033488ULL},
    {EventKind::kEvict, 36864, 3, "1", 248144ULL, 3873785344ULL},
    {EventKind::kEvict, 26624, 3, "1", 187824ULL, 3873597520ULL},
    {EventKind::kCapture, 112640, 3, "0", 0ULL, 3872792856ULL},
    {EventKind::kEvict, 15862, 2, "1", 123960ULL, 3875063120ULL},
    {EventKind::kEvict, 8192, 1, "1", 79248ULL, 3874983872ULL},
    {EventKind::kCapture, 113649, 3, "0", 0ULL, 3873351996ULL},
    {EventKind::kEvict, 4096, 1, "1", 55120ULL, 3875696548ULL},
    {EventKind::kCapture, 114156, 0, "0", 0ULL, 3874047484ULL},
    {EventKind::kEvict, 114156, 0, "1", 715204ULL, 3872394576ULL},
    {EventKind::kCapture, 2048, 3, "0", 0ULL, 3872427036ULL},
    {EventKind::kEvict, 112640, 1, "1", 694512ULL, 3872082124ULL},
    {EventKind::kEvict, 94292, 3, "1", 586320ULL, 3871495804ULL},
    {EventKind::kCapture, 6144, 3, "0", 0ULL, 3871576708ULL},
    {EventKind::kEvict, 113649, 3, "1", 3189860ULL, 3869062856ULL},
    {EventKind::kCapture, 14566, 3, "0", 0ULL, 3869200528ULL},
    {EventKind::kCapture, 16154, 0, "0", 0ULL, 3869470104ULL},
    {EventKind::kCapture, 18432, 3, "0", 0ULL, 3863770944ULL},
    {EventKind::kCapture, 26624, 3, "0", 0ULL, 3864652208ULL},
    {EventKind::kCapture, 38499, 3, "0", 0ULL, 3865905028ULL},
    {EventKind::kCapture, 38688, 0, "0", 0ULL, 3866182724ULL},
    {EventKind::kCapture, 45056, 3, "0", 0ULL, 3867025100ULL},
    {EventKind::kCapture, 60495, 3, "0", 0ULL, 3868715168ULL},
    {EventKind::kCapture, 61440, 3, "0", 0ULL, 3869192136ULL},
    {EventKind::kCapture, 61680, 0, "0", 0ULL, 3869589968ULL},
    {EventKind::kEvict, 16154, 0, "1", 125992ULL, 3872097752ULL},
    {EventKind::kEvict, 38688, 0, "1", 258720ULL, 3871839032ULL},
    {EventKind::kEvict, 61440, 3, "1", 392912ULL, 3871446120ULL},
    {EventKind::kCapture, 83968, 3, "0", 0ULL, 3871241008ULL},
    {EventKind::kEvict, 60495, 2, "1", 387244ULL, 3872652652ULL},
    {EventKind::kCapture, 84734, 3, "0", 0ULL, 3871426404ULL},
    {EventKind::kEvict, 45056, 1, "1", 296400ULL, 3872922876ULL},
    {EventKind::kCapture, 84861, 0, "0", 0ULL, 3871679016ULL},
    {EventKind::kEvict, 61680, 0, "1", 393920ULL, 3874349376ULL},
    {EventKind::kEvict, 84734, 3, "1", 529592ULL, 3873819784ULL},
    {EventKind::kEvict, 83968, 3, "1", 525616ULL, 3873294168ULL},
    {EventKind::kEvict, 38499, 3, "1", 257532ULL, 3873036636ULL},
    {EventKind::kEvict, 26624, 3, "1", 187824ULL, 3872848812ULL},
    {EventKind::kCapture, 106496, 3, "0", 0ULL, 3872288284ULL},
    {EventKind::kEvict, 18432, 1, "1", 139568ULL, 3874404604ULL},
    {EventKind::kCapture, 107253, 3, "0", 0ULL, 3872853952ULL},
    {EventKind::kEvict, 14566, 2, "1", 116408ULL, 3874368168ULL},
    {EventKind::kCapture, 107260, 0, "0", 0ULL, 3873399888ULL},
    {EventKind::kEvict, 107253, 3, "1", 662260ULL, 3876267836ULL},
    {EventKind::kEvict, 106496, 3, "1", 658320ULL, 3875609516ULL},
    {EventKind::kEvict, 84861, 3, "1", 530532ULL, 3875078984ULL},
    {EventKind::kEvict, 6144, 3, "1", 67184ULL, 3875011800ULL},
    {EventKind::kEvict, 2048, 3, "1", 42656ULL, 3874969144ULL},
    {EventKind::kRefused, 129024, 255, "byte_capacity", 0ULL, 0ULL},
    {EventKind::kSkipped, 129024, 255, "3", 738015136ULL, 0ULL},
    {EventKind::kEvict, 107260, 0, "1", 1091328ULL, 3873928512ULL},
    {EventKind::kCapture, 130011, 3, "0", 0ULL, 3873499172ULL},
    {EventKind::kRefused, 130018, 255, "byte_capacity", 0ULL, 0ULL},
    {EventKind::kSkipped, 130018, 255, "3", 139873320ULL, 0ULL},
    {EventKind::kEvict, 130011, 3, "1", 3646088ULL, 3868894152ULL},
    {EventKind::kCapture, 4096, 3, "0", 0ULL, 3868997216ULL},
    {EventKind::kCapture, 8192, 3, "0", 0ULL, 3869423184ULL},
    {EventKind::kCapture, 16352, 3, "0", 0ULL, 3870219048ULL},
    {EventKind::kCapture, 17421, 0, "0", 0ULL, 3870465820ULL},
    {EventKind::kCapture, 22528, 3, "0", 0ULL, 3864184340ULL},
    {EventKind::kCapture, 30720, 3, "0", 0ULL, 3865089732ULL},
    {EventKind::kCapture, 31117, 3, "0", 0ULL, 3865326384ULL},
    {EventKind::kCapture, 31201, 0, "0", 0ULL, 3865541412ULL},
    {EventKind::kCapture, 45056, 3, "0", 0ULL, 3867028028ULL},
    {EventKind::kCapture, 46356, 3, "0", 0ULL, 3867440388ULL},
    {EventKind::kCapture, 46417, 0, "0", 0ULL, 3867745136ULL},
    {EventKind::kCapture, 53102, 3, "0", 0ULL, 3868647952ULL},
    {EventKind::kCapture, 61440, 3, "0", 0ULL, 3869759576ULL},
    {EventKind::kCapture, 63220, 0, "0", 0ULL, 3870292104ULL},
    {EventKind::kEvict, 17421, 0, "1", 133636ULL, 3872855772ULL},
    {EventKind::kEvict, 31201, 0, "1", 214356ULL, 3872641416ULL},
    {EventKind::kEvict, 46417, 0, "1", 304260ULL, 3872337156ULL},
    {EventKind::kEvict, 61440, 3, "1", 392912ULL, 3871944244ULL},
    {EventKind::kEvict, 53102, 3, "1", 343480ULL, 3871600764ULL},
    {EventKind::kCapture, 86016, 3, "0", 0ULL, 3871391812ULL},
    {EventKind::kEvict, 46356, 2, "1", 304016ULL, 3872927644ULL},
    {EventKind::kCapture, 86708, 3, "0", 0ULL, 3871671612ULL},
    {EventKind::kEvict, 45056, 1, "1", 296400ULL, 3872706388ULL},
    {EventKind::kCapture, 86769, 0, "0", 0ULL, 3871917304ULL},
    {EventKind::kEvict, 63220, 0, "1", 402880ULL, 3874387256ULL},
    {EventKind::kEvict, 86708, 3, "1", 541360ULL, 3873845896ULL},
    {EventKind::kEvict, 86016, 3, "1", 537680ULL, 3873308216ULL},
    {EventKind::kEvict, 31117, 3, "1", 214020ULL, 3873094196ULL},
    {EventKind::kEvict, 30720, 3, "1", 211952ULL, 3872882244ULL},
    {EventKind::kCapture, 104448, 3, "0", 0ULL, 3872172964ULL},
    {EventKind::kEvict, 22528, 1, "1", 163696ULL, 3874254748ULL},
    {EventKind::kCapture, 105480, 3, "0", 0ULL, 3872750372ULL},
    {EventKind::kEvict, 16352, 2, "1", 126784ULL, 3874842268ULL},
    {EventKind::kCapture, 105756, 0, "0", 0ULL, 3873297164ULL},
    {EventKind::kEvict, 105756, 0, "1", 659112ULL, 3871912316ULL},
    {EventKind::kEvict, 104448, 1, "1", 646256ULL, 3871266060ULL},
    {EventKind::kCapture, 4096, 3, "0", 0ULL, 3871369040ULL},
    {EventKind::kEvict, 86769, 3, "1", 541604ULL, 3871210388ULL},
    {EventKind::kEvict, 8192, 1, "1", 79248ULL, 3871131140ULL},
    {EventKind::kCapture, 8192, 3, "0", 0ULL, 3871174156ULL},
    {EventKind::kEvict, 4096, 1, "1", 55120ULL, 3871896404ULL},
    {EventKind::kEvict, 105480, 3, "1", 2968604ULL, 3868927800ULL},
    {EventKind::kCapture, 17556, 3, "0", 0ULL, 3869067904ULL},
    {EventKind::kCapture, 18324, 0, "0", 0ULL, 3869264984ULL},
    {EventKind::kCapture, 20480, 3, "0", 0ULL, 3864011216ULL},
    {EventKind::kCapture, 30612, 3, "0", 0ULL, 3865063976ULL},
    {EventKind::kCapture, 34011, 3, "0", 0ULL, 3865583700ULL},
    {EventKind::kCapture, 43218, 0, "0", 0ULL, 3866648972ULL},
    {EventKind::kCapture, 61440, 3, "0", 0ULL, 3868598452ULL},
    {EventKind::kCapture, 62108, 3, "0", 0ULL, 3869037212ULL},
    {EventKind::kCapture, 62158, 0, "0", 0ULL, 3869434308ULL},
    {EventKind::kEvict, 18324, 0, "1", 138544ULL, 3871887068ULL},
    {EventKind::kEvict, 43218, 0, "1", 285224ULL, 3871601844ULL},
    {EventKind::kCapture, 83515, 3, "0", 0ULL, 3871352424ULL},
    {EventKind::kEvict, 62108, 2, "1", 396496ULL, 3872738544ULL},
    {EventKind::kCapture, 83968, 3, "0", 0ULL, 3871526736ULL},
    {EventKind::kEvict, 61440, 1, "1", 392912ULL, 3872904448ULL},
    {EventKind::kCapture, 84117, 0, "0", 0ULL, 3871663268ULL},
    {EventKind::kEvict, 62158, 0, "1", 396696ULL, 3874244820ULL},
    {EventKind::kEvict, 83968, 1, "1", 525616ULL, 3873719204ULL},
    {EventKind::kEvict, 83515, 2, "1", 522780ULL, 3873196424ULL},
    {EventKind::kEvict, 34011, 2, "1", 230972ULL, 3872965452ULL},
    {EventKind::kEvict, 30612, 1, "1", 210928ULL, 3872754524ULL},
    {EventKind::kCapture, 104513, 3, "0", 0ULL, 3872155768ULL},
    {EventKind::kEvict, 20480, 1, "1", 151632ULL, 3873596560ULL},
    {EventKind::kCapture, 104557, 0, "0", 0ULL, 3872651228ULL},
    {EventKind::kEvict, 84117, 0, "1", 526260ULL, 3875631336ULL},
    {EventKind::kEvict, 104513, 3, "1", 646564ULL, 3874984772ULL},
    {EventKind::kEvict, 17556, 3, "1", 134176ULL, 3874850596ULL},
    {EventKind::kEvict, 8192, 3, "1", 79248ULL, 3874771348ULL},
    {EventKind::kEvict, 4096, 3, "1", 55120ULL, 3874716228ULL},
    {EventKind::kCapture, 126976, 3, "0", 0ULL, 3873892916ULL},
    {EventKind::kEvict, 104557, 0, "1", 646740ULL, 3875901968ULL},
    {EventKind::kCapture, 127239, 3, "0", 0ULL, 3874048228ULL},
    {EventKind::kRefused, 127300, 255, "byte_capacity", 0ULL, 0ULL},
    {EventKind::kSkipped, 127300, 255, "3", 135110064ULL, 0ULL},
    {EventKind::kEvict, 126976, 1, "1", 778960ULL, 3872335632ULL},
    {EventKind::kEvict, 127239, 3, "1", 3574184ULL, 3868761448ULL},
    {EventKind::kCapture, 4096, 3, "0", 0ULL, 3868864512ULL},
    {EventKind::kCapture, 8192, 3, "0", 0ULL, 3869290480ULL},
    {EventKind::kCapture, 16565, 3, "0", 0ULL, 3870115636ULL},
    {EventKind::kCapture, 16572, 0, "0", 0ULL, 3870243996ULL},
    {EventKind::kCapture, 28672, 3, "0", 0ULL, 3864736772ULL},
    {EventKind::kCapture, 38622, 3, "0", 0ULL, 3865817708ULL},
    {EventKind::kCapture, 39016, 0, "0", 0ULL, 3866125092ULL},
    {EventKind::kCapture, 43008, 3, "0", 0ULL, 3866753324ULL},
    {EventKind::kCapture, 59392, 3, "0", 0ULL, 3868521052ULL},
    {EventKind::kCapture, 60642, 3, "0", 0ULL, 3868999364ULL},
    {EventKind::kCapture, 61341, 0, "0", 0ULL, 3869449272ULL},
    {EventKind::kEvict, 16572, 0, "1", 128304ULL, 3871967864ULL},
    {EventKind::kEvict, 39016, 0, "1", 260672ULL, 3871707192ULL},
    {EventKind::kCapture, 83968, 3, "0", 0ULL, 3871517768ULL},
    {EventKind::kEvict, 60642, 2, "1", 387832ULL, 3872914776ULL},
    {EventKind::kCapture, 84436, 3, "0", 0ULL, 3871681104ULL},
    {EventKind::kEvict, 59392, 1, "1", 380848ULL, 3873087896ULL},
    {EventKind::kCapture, 84558, 0, "0", 0ULL, 3871847584ULL},
    {EventKind::kEvict, 61341, 0, "1", 391924ULL, 3874528796ULL},
    {EventKind::kEvict, 84436, 3, "1", 527968ULL, 3874000828ULL},
    {EventKind::kEvict, 83968, 3, "1", 525616ULL, 3873475212ULL},
    {EventKind::kEvict, 43008, 3, "1", 284336ULL, 3873190876ULL},
    {EventKind::kEvict, 38622, 3, "1", 258024ULL, 3872932852ULL},
    {EventKind::kCapture, 106496, 3, "0", 0ULL, 3872383356ULL},
    {EventKind::kEvict, 28672, 1, "1", 199888ULL, 3874437788ULL},
    {EventKind::kCapture, 107057, 3, "0", 0ULL, 3872886352ULL},
    {EventKind::kEvict, 16565, 2, "1", 128276ULL, 3874388396ULL},
    {EventKind::kCapture, 107222, 0, "0", 0ULL, 3873421532ULL},
    {EventKind::kEvict, 107222, 0, "1", 662796ULL, 3872020032ULL},
    {EventKind::kEvict, 106496, 1, "1", 658320ULL, 3871361712ULL},
    {EventKind::kCapture, 4096, 3, "0", 0ULL, 3871464824ULL},
    {EventKind::kEvict, 84558, 3, "1", 528888ULL, 3871439816ULL},
    {EventKind::kEvict, 8192, 1, "1", 79248ULL, 3871360568ULL},
    {EventKind::kCapture, 10240, 3, "0", 0ULL, 3871468080ULL},
    {EventKind::kEvict, 4096, 1, "1", 55120ULL, 3872380200ULL},
    {EventKind::kEvict, 107057, 3, "1", 3010752ULL, 3869369448ULL},
    {EventKind::kCapture, 22160, 3, "0", 0ULL, 3869559056ULL},
    {EventKind::kCapture, 22167, 0, "0", 0ULL, 3869720340ULL},
    {EventKind::kCapture, 30720, 3, "0", 0ULL, 3865000108ULL},
    {EventKind::kCapture, 43008, 3, "0", 0ULL, 3866324604ULL},
    {EventKind::kCapture, 44893, 3, "0", 0ULL, 3866767712ULL},
    {EventKind::kCapture, 44900, 0, "0", 0ULL, 3867062952ULL},
    {EventKind::kCapture, 51200, 3, "0", 0ULL, 3867940976ULL},
    {EventKind::kCapture, 69632, 3, "0", 0ULL, 3869942384ULL},
    {EventKind::kCapture, 69880, 3, "0", 0ULL, 3870388568ULL},
    {EventKind::kCapture, 69887, 0, "0", 0ULL, 3870830860ULL},
    {EventKind::kEvict, 22167, 0, "1", 161228ULL, 3873227760ULL},
    {EventKind::kEvict, 44900, 0, "1", 295184ULL, 3872932576ULL},
    {EventKind::kEvict, 69880, 3, "1", 442208ULL, 3872490368ULL},
    {EventKind::kCapture, 88064, 3, "0", 0ULL, 3872038192ULL},
    {EventKind::kEvict, 69632, 1, "1", 441168ULL, 3873539208ULL},
    {EventKind::kEvict, 51200, 1, "1", 332592ULL, 3873206616ULL},
    {EventKind::kCapture, 89820, 3, "0", 0ULL, 3871953504ULL},
    {EventKind::kEvict, 44893, 2, "1", 295156ULL, 3873034316ULL},
    {EventKind::kCapture, 89827, 0, "0", 0ULL, 3872218048ULL},
    {EventKind::kEvict, 69887, 0, "1", 442236ULL, 3875134220ULL},
    {EventKind::kEvict, 89820, 3, "1", 559616ULL, 3874574604ULL},
    {EventKind::kEvict, 88064, 3, "1", 549744ULL, 3874024860ULL},
    {EventKind::kEvict, 43008, 3, "1", 284336ULL, 3873740524ULL},
    {EventKind::kEvict, 30720, 3, "1", 211952ULL, 3873528572ULL},
    {EventKind::kEvict, 22160, 3, "1", 161200ULL, 3873367372ULL},
    {EventKind::kCapture, 114688, 3, "0", 0ULL, 3872839708ULL},
    {EventKind::kEvict, 10240, 1, "1", 91312ULL, 3875216252ULL},
    {EventKind::kCapture, 116089, 3, "0", 0ULL, 3873572144ULL},
    {EventKind::kEvict, 4096, 1, "1", 55120ULL, 3875277808ULL},
    {EventKind::kCapture, 116096, 0, "0", 0ULL, 3874231704ULL},
}};
inline constexpr std::array<Request, 46> kDeepRequests = {{
    {"more-ernie-row1086--p00", 0, -1, 87339, 0, 0, 87339, 78, 0ULL,
     Reuse::kNone, 0, 4, 200},  // r2
    {"more-ernie-row1086--p01", 0, -1, 97460, 87339, 87339, 10121, 161, 2531356492ULL,
     Reuse::kRam, 4, 3, 200},  // r3
    {"more-ernie-row1086--p02", 0, -1, 107031, 97460, 97460, 9571, 38, 2809281392ULL,
     Reuse::kRam, 7, 7, 200},  // r4
    {"more-ernie-row1086--p03", 0, -1, 117318, 107031, 107069, 10249, 109, 0ULL,
     Reuse::kLive, 14, 10, 200},  // r5
    {"more-ernie-row1086--p04", 0, -1, 126714, 117318, 117318, 9396, 106, 3354580664ULL,
     Reuse::kRam, 24, 8, 200},  // r6
    {"more-ernie-row1086--p05", 0, -1, 136579, 126714, 126707, 9872, 88, 3612401772ULL,
     Reuse::kRam, 32, 8, 200},  // r7
    {"more-ernie-row204--p00", 1, -1, 17436, 0, 0, 17436, 234, 0ULL,
     Reuse::kNone, 40, 4, 200},  // r8
    {"more-ernie-row204--p01", 1, -1, 45709, 17436, 17436, 28273, 157, 611822352ULL,
     Reuse::kRam, 44, 4, 200},  // r9
    {"more-ernie-row204--p02", 1, -1, 46920, 45709, 45709, 1211, 256, 1388198356ULL,
     Reuse::kRam, 48, 2, 200},  // r10
    {"more-ernie-row204--p03", 1, -1, 143689, 46920, 46920, 96769, 256, 1421452992ULL,
     Reuse::kRam, 50, 14, 200},  // r11
    {"more-ernie-row204--p04", 1, -1, 144214, 143689, 106496, 37718, 256, 3057409952ULL,
     Reuse::kRam, 64, 10, 200},  // r12
    {"more-ernie-row204--p05", 1, -1, 153742, 144214, 0, 153742, 256, 0ULL,
     Reuse::kNone, 74, 6, 200},  // r13
    {"more-ernie-row304--p00", 2, -1, 18302, 0, 0, 18302, 105, 0ULL,
     Reuse::kNone, 80, 4, 200},  // r14
    {"more-ernie-row304--p01", 2, -1, 37738, 18302, 18302, 19436, 78, 635601304ULL,
     Reuse::kRam, 84, 5, 200},  // r15
    {"more-ernie-row304--p02", 2, -1, 56475, 37738, 37738, 18737, 138, 1169313864ULL,
     Reuse::kRam, 89, 6, 200},  // r16
    {"more-ernie-row304--p03", 2, -1, 75381, 56475, 56475, 18906, 167, 1683831052ULL,
     Reuse::kRam, 95, 5, 200},  // r17
    {"more-ernie-row304--p04", 2, -1, 94292, 75381, 75381, 18911, 109, 2202991476ULL,
     Reuse::kRam, 100, 8, 200},  // r18
    {"more-ernie-row304--p05", 2, -1, 114156, 94292, 94292, 19864, 256, 2722288112ULL,
     Reuse::kRam, 108, 10, 200},  // r19
    {"more-ernie-row3427--p00", 3, -1, 16154, 0, 0, 16154, 128, 0ULL,
     Reuse::kNone, 118, 8, 200},  // r20
    {"more-ernie-row3427--p01", 3, -1, 38688, 16154, 16154, 22534, 58, 576617224ULL,
     Reuse::kRam, 126, 4, 200},  // r21
    {"more-ernie-row3427--p02", 3, -1, 61680, 38688, 38688, 22992, 157, 1195402272ULL,
     Reuse::kRam, 130, 4, 200},  // r22
    {"more-ernie-row3427--p03", 3, -1, 84861, 61680, 61680, 23181, 104, 1826762592ULL,
     Reuse::kRam, 134, 8, 200},  // r23
    {"more-ernie-row3427--p04", 3, -1, 107260, 84861, 84861, 22399, 256, 2463312276ULL,
     Reuse::kRam, 142, 10, 200},  // r24
    {"more-ernie-row3427--p05", 3, -1, 130018, 107260, 107260, 22758, 256, 3078389392ULL,
     Reuse::kRam, 152, 11, 200},  // r25
    {"more-ernie-row343--p00", 4, -1, 17421, 0, 0, 17421, 167, 0ULL,
     Reuse::kNone, 163, 5, 200},  // r26
    {"more-ernie-row343--p01", 4, -1, 31201, 17421, 17421, 13780, 189, 611409876ULL,
     Reuse::kRam, 168, 4, 200},  // r27
    {"more-ernie-row343--p02", 4, -1, 46417, 31201, 31201, 15216, 100, 989808676ULL,
     Reuse::kRam, 172, 3, 200},  // r28
    {"more-ernie-row343--p03", 4, -1, 63220, 46417, 46417, 16803, 103, 1407640036ULL,
     Reuse::kRam, 175, 3, 200},  // r29
    {"more-ernie-row343--p04", 4, -1, 86769, 63220, 63220, 23549, 256, 1869050992ULL,
     Reuse::kRam, 178, 10, 200},  // r30
    {"more-ernie-row343--p05", 4, -1, 105756, 86769, 86769, 18987, 164, 2515705956ULL,
     Reuse::kRam, 188, 10, 200},  // r31
    {"more-ernie-row393--p00", 5, -1, 18324, 0, 0, 18324, 64, 0ULL,
     Reuse::kNone, 198, 10, 200},  // r32
    {"more-ernie-row393--p01", 5, -1, 43218, 18324, 18324, 24894, 256, 636206832ULL,
     Reuse::kRam, 208, 4, 200},  // r33
    {"more-ernie-row393--p02", 5, -1, 62158, 43218, 43218, 18940, 256, 1319794664ULL,
     Reuse::kRam, 212, 3, 200},  // r34
    {"more-ernie-row393--p03", 5, -1, 84117, 62158, 62158, 21959, 111, 1839887064ULL,
     Reuse::kRam, 215, 7, 200},  // r35
    {"more-ernie-row393--p04", 5, -1, 104557, 84117, 84117, 20440, 72, 2442882036ULL,
     Reuse::kRam, 222, 8, 200},  // r36
    {"more-ernie-row393--p05", 5, -1, 127300, 104557, 104557, 22743, 256, 3004164436ULL,
     Reuse::kRam, 230, 10, 200},  // r37
    {"more-ernie-row872--p00", 6, -1, 16572, 0, 0, 16572, 61, 0ULL,
     Reuse::kNone, 240, 6, 200},  // r38
    {"more-ernie-row872--p01", 6, -1, 39016, 16572, 16572, 22444, 256, 588096912ULL,
     Reuse::kRam, 246, 3, 200},  // r39
    {"more-ernie-row872--p02", 6, -1, 61341, 39016, 39016, 22325, 74, 1204409152ULL,
     Reuse::kRam, 249, 4, 200},  // r40
    {"more-ernie-row872--p03", 6, -1, 84558, 61341, 61341, 23217, 53, 1817453076ULL,
     Reuse::kRam, 253, 7, 200},  // r41
    {"more-ernie-row872--p04", 6, -1, 107222, 84558, 84558, 22664, 109, 2454991064ULL,
     Reuse::kRam, 260, 10, 200},  // r42
    {"more-nvidia-33102--p00", 7, -1, 22167, 0, 0, 22167, 72, 0ULL,
     Reuse::kNone, 270, 10, 200},  // r43
    {"more-nvidia-33102--p01", 7, -1, 44900, 22167, 22167, 22733, 56, 741733372ULL,
     Reuse::kRam, 280, 4, 200},  // r44
    {"more-nvidia-33102--p02", 7, -1, 69887, 44900, 44900, 24987, 94, 1365983792ULL,
     Reuse::kRam, 284, 4, 200},  // r45
    {"more-nvidia-33102--p03", 7, -1, 89827, 69887, 69887, 19940, 187, 2052124572ULL,
     Reuse::kRam, 288, 9, 200},  // r46
    {"more-nvidia-33102--p04", 7, -1, 116096, 89827, 89827, 26269, 256, 2599676972ULL,
     Reuse::kRam, 297, 11, 200},  // r47
}};

// server.err of recorded run pf-merge-fork4g-c0, arm C0
// SHA256 f774d2d2be4683741412cc9ee33c06edec87517477077b1ddf594ecafdc722a5
inline constexpr std::array<Event, 184> kForkEvents = {{
    {EventKind::kCapture, 2048, 3, "0", 0ULL, 2288988776ULL},
    {EventKind::kCapture, 6144, 3, "0", 0ULL, 2420556816ULL},
    {EventKind::kCapture, 14566, 3, "0", 0ULL, 2683645568ULL},
    {EventKind::kCapture, 16154, 0, "0", 0ULL, 3077327752ULL},
    {EventKind::kCapture, 18432, 3, "0", 0ULL, 3208804160ULL},
    {EventKind::kCapture, 26624, 3, "0", 0ULL, 3471960496ULL},
    {EventKind::kCapture, 38499, 3, "0", 0ULL, 3865905028ULL},
    {EventKind::kCapture, 38688, 0, "0", 0ULL, 3866182724ULL},
    {EventKind::kCapture, 45056, 3, "0", 0ULL, 3867025100ULL},
    {EventKind::kCapture, 60495, 3, "0", 0ULL, 3868715168ULL},
    {EventKind::kCapture, 61440, 3, "0", 0ULL, 3869192136ULL},
    {EventKind::kCapture, 61680, 0, "0", 0ULL, 3869589968ULL},
    {EventKind::kCapture, 61812, 3, "0", 0ULL, 3870003368ULL},
    {EventKind::kCapture, 61819, 0, "0", 0ULL, 3870398332ULL},
    {EventKind::kEvict, 16154, 0, "1", 125992ULL, 3872904264ULL},
    {EventKind::kEvict, 38688, 0, "1", 258720ULL, 3872645544ULL},
    {EventKind::kEvict, 61819, 0, "1", 394936ULL, 3872250608ULL},
    {EventKind::kEvict, 61440, 1, "1", 392912ULL, 3871857696ULL},
    {EventKind::kEvict, 60495, 2, "1", 387244ULL, 3871470452ULL},
    {EventKind::kCapture, 83968, 3, "0", 0ULL, 3871265032ULL},
    {EventKind::kEvict, 45056, 1, "1", 296400ULL, 3872767520ULL},
    {EventKind::kCapture, 84734, 3, "0", 0ULL, 3871541272ULL},
    {EventKind::kEvict, 38499, 2, "1", 257532ULL, 3873076612ULL},
    {EventKind::kCapture, 84861, 0, "0", 0ULL, 3871832752ULL},
    {EventKind::kEvict, 61680, 0, "1", 393920ULL, 3874503112ULL},
    {EventKind::kEvict, 84734, 3, "1", 529592ULL, 3873973520ULL},
    {EventKind::kEvict, 83968, 3, "1", 525616ULL, 3873447904ULL},
    {EventKind::kEvict, 26624, 1, "1", 187824ULL, 3873260080ULL},
    {EventKind::kEvict, 18432, 1, "1", 139568ULL, 3873120512ULL},
    {EventKind::kCapture, 106496, 3, "0", 0ULL, 3872559984ULL},
    {EventKind::kEvict, 14566, 2, "1", 116408ULL, 3874699464ULL},
    {EventKind::kCapture, 107253, 3, "0", 0ULL, 3873148812ULL},
    {EventKind::kEvict, 6144, 1, "1", 67184ULL, 3874712252ULL},
    {EventKind::kCapture, 107260, 0, "0", 0ULL, 3873743972ULL},
    {EventKind::kEvict, 84861, 0, "1", 530532ULL, 3875481880ULL},
    {EventKind::kEvict, 107253, 2, "1", 662260ULL, 3874819620ULL},
    {EventKind::kCapture, 107541, 3, "0", 0ULL, 3873261472ULL},
    {EventKind::kEvict, 106496, 1, "1", 658320ULL, 3874239832ULL},
    {EventKind::kCapture, 107548, 0, "0", 0ULL, 3873267720ULL},
    {EventKind::kEvict, 107548, 0, "1", 664540ULL, 3876129020ULL},
    {EventKind::kEvict, 2048, 1, "1", 42656ULL, 3876086364ULL},
    {EventKind::kEvict, 61812, 3, "1", 399204ULL, 3875687160ULL},
    {EventKind::kEvict, 107541, 3, "1", 675792ULL, 3875011368ULL},
    {EventKind::kRefused, 129024, 255, "byte_capacity", 0ULL, 0ULL},
    {EventKind::kSkipped, 129024, 255, "3", 738015136ULL, 0ULL},
    {EventKind::kEvict, 107260, 0, "1", 1091328ULL, 3873970736ULL},
    {EventKind::kCapture, 130011, 3, "0", 0ULL, 3873541396ULL},
    {EventKind::kRefused, 130018, 255, "byte_capacity", 0ULL, 0ULL},
    {EventKind::kSkipped, 130018, 255, "3", 139873320ULL, 0ULL},
    {EventKind::kEvict, 130011, 3, "1", 3646088ULL, 3868894152ULL},
    {EventKind::kCapture, 4096, 3, "0", 0ULL, 3868997216ULL},
    {EventKind::kCapture, 8192, 3, "0", 0ULL, 3869423184ULL},
    {EventKind::kCapture, 17556, 3, "0", 0ULL, 3870340656ULL},
    {EventKind::kCapture, 18324, 0, "0", 0ULL, 3870537736ULL},
    {EventKind::kCapture, 20480, 3, "0", 0ULL, 3864011216ULL},
    {EventKind::kCapture, 30612, 3, "0", 0ULL, 3865063976ULL},
    {EventKind::kCapture, 34011, 3, "0", 0ULL, 3865583700ULL},
    {EventKind::kCapture, 43218, 0, "0", 0ULL, 3866648972ULL},
    {EventKind::kCapture, 61440, 3, "0", 0ULL, 3868598452ULL},
    {EventKind::kCapture, 62108, 3, "0", 0ULL, 3869037212ULL},
    {EventKind::kCapture, 62158, 0, "0", 0ULL, 3869434308ULL},
    {EventKind::kCapture, 62550, 3, "0", 0ULL, 3869880340ULL},
    {EventKind::kCapture, 62557, 0, "0", 0ULL, 3870279760ULL},
    {EventKind::kEvict, 18324, 0, "1", 138544ULL, 3872727708ULL},
    {EventKind::kEvict, 43218, 0, "1", 285224ULL, 3872442484ULL},
    {EventKind::kEvict, 62557, 0, "1", 399392ULL, 3872043092ULL},
    {EventKind::kEvict, 62108, 2, "1", 396496ULL, 3871646596ULL},
    {EventKind::kCapture, 83515, 3, "0", 0ULL, 3871395828ULL},
    {EventKind::kEvict, 61440, 1, "1", 392912ULL, 3872785532ULL},
    {EventKind::kCapture, 83968, 3, "0", 0ULL, 3871573724ULL},
    {EventKind::kEvict, 34011, 2, "1", 230972ULL, 3873113376ULL},
    {EventKind::kCapture, 84117, 0, "0", 0ULL, 3871872196ULL},
    {EventKind::kEvict, 62158, 0, "1", 396696ULL, 3874453748ULL},
    {EventKind::kEvict, 83968, 1, "1", 525616ULL, 3873928132ULL},
    {EventKind::kEvict, 83515, 2, "1", 522780ULL, 3873405352ULL},
    {EventKind::kEvict, 30612, 1, "1", 210928ULL, 3873194424ULL},
    {EventKind::kEvict, 20480, 1, "1", 151632ULL, 3873042792ULL},
    {EventKind::kCapture, 104513, 3, "0", 0ULL, 3872444036ULL},
    {EventKind::kEvict, 17556, 2, "1", 134176ULL, 3873902284ULL},
    {EventKind::kCapture, 104557, 0, "0", 0ULL, 3872956952ULL},
    {EventKind::kEvict, 84117, 0, "1", 526260ULL, 3874629092ULL},
    {EventKind::kCapture, 104794, 3, "0", 0ULL, 3873098172ULL},
    {EventKind::kEvict, 104513, 2, "1", 646564ULL, 3874047496ULL},
    {EventKind::kCapture, 104801, 0, "0", 0ULL, 3873099812ULL},
    {EventKind::kEvict, 104801, 0, "1", 648176ULL, 3875955732ULL},
    {EventKind::kEvict, 8192, 1, "1", 79248ULL, 3875876484ULL},
    {EventKind::kEvict, 4096, 1, "1", 55120ULL, 3875821364ULL},
    {EventKind::kEvict, 62550, 3, "1", 409740ULL, 3875411624ULL},
    {EventKind::kEvict, 104794, 3, "1", 653172ULL, 3874758452ULL},
    {EventKind::kCapture, 126976, 3, "0", 0ULL, 3873935140ULL},
    {EventKind::kEvict, 104557, 0, "1", 646740ULL, 3875944192ULL},
    {EventKind::kCapture, 127239, 3, "0", 0ULL, 3874090452ULL},
    {EventKind::kRefused, 127300, 255, "byte_capacity", 0ULL, 0ULL},
    {EventKind::kSkipped, 127300, 255, "3", 135110064ULL, 0ULL},
    {EventKind::kEvict, 126976, 1, "1", 778960ULL, 3872335632ULL},
    {EventKind::kEvict, 127239, 3, "1", 3574184ULL, 3868761448ULL},
    {EventKind::kCapture, 4096, 3, "0", 0ULL, 3868864512ULL},
    {EventKind::kCapture, 8192, 3, "0", 0ULL, 3869290480ULL},
    {EventKind::kCapture, 15862, 3, "0", 0ULL, 3870044672ULL},
    {EventKind::kCapture, 18302, 0, "0", 0ULL, 3870394552ULL},
    {EventKind::kCapture, 26624, 3, "0", 0ULL, 3864557184ULL},
    {EventKind::kCapture, 36864, 3, "0", 0ULL, 3865672128ULL},
    {EventKind::kCapture, 37731, 3, "0", 0ULL, 3865986404ULL},
    {EventKind::kCapture, 37738, 0, "0", 0ULL, 3866239444ULL},
    {EventKind::kCapture, 40960, 3, "0", 0ULL, 3866797060ULL},
    {EventKind::kCapture, 55296, 3, "0", 0ULL, 3868367300ULL},
    {EventKind::kCapture, 56128, 3, "0", 0ULL, 3868789732ULL},
    {EventKind::kCapture, 56475, 0, "0", 0ULL, 3869182024ULL},
    {EventKind::kCapture, 56642, 3, "0", 0ULL, 3869565344ULL},
    {EventKind::kCapture, 56649, 0, "0", 0ULL, 3869929948ULL},
    {EventKind::kEvict, 18302, 0, "1", 138456ULL, 3872017076ULL},
    {EventKind::kEvict, 37738, 0, "1", 252984ULL, 3871764092ULL},
    {EventKind::kEvict, 56649, 0, "1", 364576ULL, 3871399516ULL},
    {EventKind::kCapture, 73728, 3, "0", 0ULL, 3871107116ULL},
    {EventKind::kEvict, 56128, 2, "1", 361392ULL, 3872396164ULL},
    {EventKind::kCapture, 75374, 3, "0", 0ULL, 3871349028ULL},
    {EventKind::kEvict, 55296, 1, "1", 356720ULL, 3872157844ULL},
    {EventKind::kCapture, 75381, 0, "0", 0ULL, 3871467120ULL},
    {EventKind::kEvict, 56475, 0, "1", 363420ULL, 3873816884ULL},
    {EventKind::kEvict, 75374, 2, "1", 474728ULL, 3873342156ULL},
    {EventKind::kEvict, 73728, 1, "1", 465296ULL, 3872876860ULL},
    {EventKind::kEvict, 40960, 1, "1", 272272ULL, 3872604588ULL},
    {EventKind::kEvict, 37731, 2, "1", 252956ULL, 3872351632ULL},
    {EventKind::kCapture, 94285, 3, "0", 0ULL, 3871832316ULL},
    {EventKind::kEvict, 36864, 1, "1", 248144ULL, 3873026716ULL},
    {EventKind::kCapture, 94292, 0, "0", 0ULL, 3872170548ULL},
    {EventKind::kEvict, 75381, 0, "1", 474756ULL, 3873686192ULL},
    {EventKind::kCapture, 94524, 3, "0", 0ULL, 3872302792ULL},
    {EventKind::kEvict, 94285, 2, "1", 586292ULL, 3873162468ULL},
    {EventKind::kCapture, 94531, 0, "0", 0ULL, 3872304264ULL},
    {EventKind::kEvict, 94531, 0, "1", 587736ULL, 3874771028ULL},
    {EventKind::kEvict, 26624, 1, "1", 187824ULL, 3874583204ULL},
    {EventKind::kEvict, 15862, 2, "1", 123960ULL, 3874459244ULL},
    {EventKind::kEvict, 8192, 1, "1", 79248ULL, 3874379996ULL},
    {EventKind::kEvict, 4096, 1, "1", 55120ULL, 3874324876ULL},
    {EventKind::kCapture, 112640, 3, "0", 0ULL, 3873519504ULL},
    {EventKind::kEvict, 94292, 0, "1", 586320ULL, 3875327408ULL},
    {EventKind::kCapture, 113649, 3, "0", 0ULL, 3873695532ULL},
    {EventKind::kRefused, 114156, 255, "byte_capacity", 0ULL, 0ULL},
    {EventKind::kSkipped, 114156, 255, "3", 154208848ULL, 0ULL},
    {EventKind::kEvict, 112640, 1, "1", 694512ULL, 3872200464ULL},
    {EventKind::kEvict, 56642, 3, "1", 368844ULL, 3871831620ULL},
    {EventKind::kCapture, 4096, 3, "0", 0ULL, 3871934732ULL},
    {EventKind::kEvict, 94524, 3, "1", 592004ULL, 3871846608ULL},
    {EventKind::kEvict, 113649, 3, "1", 3189992ULL, 3868656616ULL},
    {EventKind::kCapture, 10240, 3, "0", 0ULL, 3868764128ULL},
    {EventKind::kCapture, 22160, 3, "0", 0ULL, 3869920976ULL},
    {EventKind::kCapture, 22167, 0, "0", 0ULL, 3870082260ULL},
    {EventKind::kCapture, 30720, 3, "0", 0ULL, 3865000108ULL},
    {EventKind::kCapture, 43008, 3, "0", 0ULL, 3866324604ULL},
    {EventKind::kCapture, 44893, 3, "0", 0ULL, 3866767712ULL},
    {EventKind::kCapture, 44900, 0, "0", 0ULL, 3867062952ULL},
    {EventKind::kCapture, 51200, 3, "0", 0ULL, 3867940976ULL},
    {EventKind::kCapture, 69632, 3, "0", 0ULL, 3869942384ULL},
    {EventKind::kCapture, 69880, 3, "0", 0ULL, 3870388568ULL},
    {EventKind::kCapture, 69887, 0, "0", 0ULL, 3870830860ULL},
    {EventKind::kCapture, 70001, 3, "0", 0ULL, 3871292360ULL},
    {EventKind::kEvict, 22167, 0, "1", 161228ULL, 3872218292ULL},
    {EventKind::kCapture, 70008, 0, "0", 0ULL, 3871574340ULL},
    {EventKind::kEvict, 44900, 0, "1", 295184ULL, 3873835504ULL},
    {EventKind::kEvict, 70008, 0, "1", 443180ULL, 3873392324ULL},
    {EventKind::kEvict, 69880, 2, "1", 442208ULL, 3872950116ULL},
    {EventKind::kEvict, 69632, 1, "1", 441168ULL, 3872508948ULL},
    {EventKind::kEvict, 51200, 1, "1", 332592ULL, 3872176356ULL},
    {EventKind::kCapture, 88064, 3, "0", 0ULL, 3871723944ULL},
    {EventKind::kEvict, 44893, 2, "1", 295156ULL, 3873370972ULL},
    {EventKind::kCapture, 89820, 3, "0", 0ULL, 3872117860ULL},
    {EventKind::kEvict, 43008, 1, "1", 284336ULL, 3873209492ULL},
    {EventKind::kCapture, 89827, 0, "0", 0ULL, 3872393224ULL},
    {EventKind::kEvict, 69887, 0, "1", 442236ULL, 3875309396ULL},
    {EventKind::kEvict, 89820, 3, "1", 559616ULL, 3874749780ULL},
    {EventKind::kEvict, 88064, 3, "1", 549744ULL, 3874200036ULL},
    {EventKind::kEvict, 30720, 1, "1", 211952ULL, 3873988084ULL},
    {EventKind::kEvict, 22160, 2, "1", 161200ULL, 3873826884ULL},
    {EventKind::kEvict, 10240, 1, "1", 91312ULL, 3873735572ULL},
    {EventKind::kCapture, 114688, 3, "0", 0ULL, 3873207908ULL},
    {EventKind::kEvict, 4096, 1, "1", 55120ULL, 3875620644ULL},
    {EventKind::kCapture, 116089, 3, "0", 0ULL, 3873976536ULL},
    {EventKind::kRefused, 116096, 255, "byte_capacity", 0ULL, 0ULL},
    {EventKind::kSkipped, 116096, 255, "3", 136757664ULL, 0ULL},
    {EventKind::kEvict, 89827, 0, "1", 559644ULL, 3875178348ULL},
    {EventKind::kCapture, 116180, 3, "0", 0ULL, 3874132580ULL},
    {EventKind::kEvict, 114688, 1, "1", 706576ULL, 3875187516ULL},
    {EventKind::kCapture, 116187, 0, "0", 0ULL, 3874141048ULL},
}};
inline constexpr std::array<Request, 31> kForkRequests = {{
    {"more-ernie-row3427--p00", 3, -1, 16154, 0, 0, 16154, 128, 0ULL,
     Reuse::kNone, 0, 4, 200},  // r2
    {"more-ernie-row3427--p01", 3, -1, 38688, 16154, 16154, 22534, 58, 576617224ULL,
     Reuse::kRam, 4, 4, 200},  // r3
    {"more-ernie-row3427--p02", 3, -1, 61680, 38688, 38688, 22992, 157, 1195402272ULL,
     Reuse::kRam, 8, 4, 200},  // r4
    {"more-ernie-row3427--fork-p02--p00", 8, 3, 61819, 61680, 61680, 139, 229, 1826762592ULL,
     Reuse::kRam, 12, 2, 200},  // r5
    {"more-ernie-row3427--p03", 3, -1, 84861, 61680, 61680, 23181, 104, 1826762592ULL,
     Reuse::kRam, 14, 10, 200},  // r6
    {"more-ernie-row3427--p04", 3, -1, 107260, 84861, 84861, 22399, 256, 2463312276ULL,
     Reuse::kRam, 24, 10, 200},  // r7
    {"more-ernie-row3427--fork-p04--p00", 9, 3, 107548, 107260, 107260, 288, 256, 3078389392ULL,
     Reuse::kRam, 34, 5, 200},  // r8
    {"more-ernie-row3427--p05", 3, -1, 130018, 107260, 107260, 22758, 256, 3078389392ULL,
     Reuse::kRam, 39, 10, 200},  // r9
    {"more-ernie-row393--p00", 5, -1, 18324, 0, 0, 18324, 64, 0ULL,
     Reuse::kNone, 49, 5, 200},  // r10
    {"more-ernie-row393--p01", 5, -1, 43218, 18324, 18324, 24894, 256, 636206832ULL,
     Reuse::kRam, 54, 4, 200},  // r11
    {"more-ernie-row393--p02", 5, -1, 62158, 43218, 43218, 18940, 256, 1319794664ULL,
     Reuse::kRam, 58, 3, 200},  // r12
    {"more-ernie-row393--fork-p02--p00", 10, 5, 62557, 62158, 62158, 399, 256, 1839887064ULL,
     Reuse::kRam, 61, 2, 200},  // r13
    {"more-ernie-row393--p03", 5, -1, 84117, 62158, 62158, 21959, 111, 1839887064ULL,
     Reuse::kRam, 63, 9, 200},  // r14
    {"more-ernie-row393--p04", 5, -1, 104557, 84117, 84117, 20440, 72, 2442882036ULL,
     Reuse::kRam, 72, 8, 200},  // r15
    {"more-ernie-row393--fork-p04--p00", 11, 5, 104801, 104557, 104557, 244, 242, 3004164436ULL,
     Reuse::kRam, 80, 4, 200},  // r16
    {"more-ernie-row393--p05", 5, -1, 127300, 104557, 104557, 22743, 256, 3004164436ULL,
     Reuse::kRam, 84, 10, 200},  // r17
    {"more-ernie-row304--p00", 2, -1, 18302, 0, 0, 18302, 105, 0ULL,
     Reuse::kNone, 94, 6, 200},  // r18
    {"more-ernie-row304--p01", 2, -1, 37738, 18302, 18302, 19436, 78, 635601304ULL,
     Reuse::kRam, 100, 4, 200},  // r19
    {"more-ernie-row304--p02", 2, -1, 56475, 37738, 37738, 18737, 138, 1169313864ULL,
     Reuse::kRam, 104, 4, 200},  // r20
    {"more-ernie-row304--fork-p02--p00", 12, 2, 56649, 56475, 56475, 174, 256, 1683831052ULL,
     Reuse::kRam, 108, 2, 200},  // r21
    {"more-ernie-row304--p03", 2, -1, 75381, 56475, 56475, 18906, 167, 1683831052ULL,
     Reuse::kRam, 110, 8, 200},  // r22
    {"more-ernie-row304--p04", 2, -1, 94292, 75381, 75381, 18911, 109, 2202991476ULL,
     Reuse::kRam, 118, 8, 200},  // r23
    {"more-ernie-row304--fork-p04--p00", 13, 2, 94531, 94292, 94292, 239, 240, 2722288112ULL,
     Reuse::kRam, 126, 4, 200},  // r24
    {"more-ernie-row304--p05", 2, -1, 114156, 94292, 94292, 19864, 256, 2722288112ULL,
     Reuse::kRam, 130, 10, 200},  // r25
    {"more-nvidia-33102--p00", 7, -1, 22167, 0, 0, 22167, 72, 0ULL,
     Reuse::kNone, 140, 8, 200},  // r26
    {"more-nvidia-33102--p01", 7, -1, 44900, 22167, 22167, 22733, 56, 741733372ULL,
     Reuse::kRam, 148, 4, 200},  // r27
    {"more-nvidia-33102--p02", 7, -1, 69887, 44900, 44900, 24987, 94, 1365983792ULL,
     Reuse::kRam, 152, 4, 200},  // r28
    {"more-nvidia-33102--fork-p02--p00", 14, 7, 70008, 69887, 69887, 121, 256, 2052124572ULL,
     Reuse::kRam, 156, 3, 200},  // r29
    {"more-nvidia-33102--p03", 7, -1, 89827, 69887, 69887, 19940, 187, 2052124572ULL,
     Reuse::kRam, 159, 10, 200},  // r30
    {"more-nvidia-33102--p04", 7, -1, 116096, 89827, 89827, 26269, 256, 2599676972ULL,
     Reuse::kRam, 169, 11, 200},  // r31
    {"more-nvidia-33102--fork-p04--p00", 15, 7, 116187, 116096, 116089, 98, 256, 3320833156ULL,
     Reuse::kRam, 180, 4, 200},  // r32
}};

// server.err of recorded run pf-merge2-fork4g, arm C0
// SHA256 60087dcadac7869b15c8836b19b0e5232592292281f6fa8294fd1c32fb0bf1a5
inline constexpr std::array<Event, 39> kFork6267Events = {{
    {EventKind::kCapture, 2048, 3, "0", 0ULL, 321927440ULL},
    {EventKind::kCapture, 6144, 3, "0", 0ULL, 584637056ULL},
    {EventKind::kCapture, 14566, 3, "0", 0ULL, 978871048ULL},
    {EventKind::kCapture, 16154, 0, "0", 0ULL, 1110279712ULL},
    {EventKind::kCapture, 18432, 3, "0", 0ULL, 1372895824ULL},
    {EventKind::kCapture, 26624, 3, "0", 0ULL, 1636059840ULL},
    {EventKind::kCapture, 38499, 3, "0", 0ULL, 2161873844ULL},
    {EventKind::kCapture, 38688, 0, "0", 0ULL, 2293289228ULL},
    {EventKind::kCapture, 45056, 3, "0", 0ULL, 2556412716ULL},
    {EventKind::kCapture, 60495, 3, "0", 0ULL, 3213804944ULL},
    {EventKind::kCapture, 61440, 3, "0", 0ULL, 3345420328ULL},
    {EventKind::kCapture, 61680, 0, "0", 0ULL, 3476955656ULL},
    {EventKind::kCapture, 61812, 3, "0", 0ULL, 3608506744ULL},
    {EventKind::kCapture, 61819, 0, "0", 0ULL, 3740039180ULL},
    {EventKind::kEvict, 16154, 0, "1", 125824ULL, 3740115480ULL},
    {EventKind::kEvict, 38688, 0, "1", 258552ULL, 3739856928ULL},
    {EventKind::kEvict, 61819, 0, "1", 394872ULL, 3739462056ULL},
    {EventKind::kEvict, 61440, 1, "1", 392744ULL, 3739069312ULL},
    {EventKind::kEvict, 60495, 2, "1", 387076ULL, 3738682236ULL},
    {EventKind::kCapture, 83968, 3, "0", 0ULL, 3871344248ULL},
    {EventKind::kEvict, 45056, 1, "1", 296232ULL, 3871212672ULL},
    {EventKind::kCapture, 84734, 3, "0", 0ULL, 3871621048ULL},
    {EventKind::kEvict, 38499, 2, "1", 257364ULL, 3871528292ULL},
    {EventKind::kCapture, 84861, 0, "0", 0ULL, 3871912848ULL},
    {EventKind::kEvict, 61680, 0, "1", 393752ULL, 3871764584ULL},
    {EventKind::kEvict, 84734, 3, "1", 529424ULL, 3871235160ULL},
    {EventKind::kEvict, 83968, 3, "1", 525448ULL, 3870709712ULL},
    {EventKind::kEvict, 26624, 1, "1", 187656ULL, 3870522056ULL},
    {EventKind::kEvict, 18432, 1, "1", 139400ULL, 3870382656ULL},
    {EventKind::kCapture, 106496, 3, "0", 0ULL, 3872661480ULL},
    {EventKind::kEvict, 14566, 2, "1", 116240ULL, 3872752488ULL},
    {EventKind::kCapture, 107253, 3, "0", 0ULL, 3873250868ULL},
    {EventKind::kEvict, 6144, 1, "1", 67016ULL, 3873390188ULL},
    {EventKind::kCapture, 107260, 0, "0", 0ULL, 3873846132ULL},
    {EventKind::kEvict, 84861, 3, "1", 530364ULL, 3873524248ULL},
    {EventKind::kEvict, 107253, 3, "1", 662092ULL, 3872862156ULL},
    {EventKind::kCapture, 107541, 3, "0", 0ULL, 3873364440ULL},
    {EventKind::kEvict, 106496, 1, "1", 658152ULL, 3872913696ULL},
    {EventKind::kCapture, 107548, 0, "0", 0ULL, 3873370792ULL},
}};
inline constexpr std::array<Request, 8> kFork6267Requests = {{
    {"more-ernie-row3427--p00", 3, -1, 16154, 0, 0, 16154, 128, 0ULL,
     Reuse::kNone, 0, 4, 200},  // r2
    {"more-ernie-row3427--p01", 3, -1, 38688, 16154, 16154, 22534, 58, 576617224ULL,
     Reuse::kRam, 4, 4, 200},  // r3
    {"more-ernie-row3427--p02", 3, -1, 61680, 38688, 38688, 22992, 157, 1195402272ULL,
     Reuse::kRam, 8, 4, 200},  // r4
    {"more-ernie-row3427--fork-p02--p00", 8, 3, 61819, 61680, 61680, 139, 229, 1826762592ULL,
     Reuse::kRam, 12, 2, 200},  // r5
    {"more-ernie-row3427--p03", 3, -1, 84861, 61680, 61680, 23181, 104, 1826762592ULL,
     Reuse::kRam, 14, 10, 200},  // r6
    {"more-ernie-row3427--p04", 3, -1, 107260, 84861, 84861, 22399, 256, 2463312276ULL,
     Reuse::kRam, 24, 10, 200},  // r7
    {"more-ernie-row3427--fork-p04--p00", 9, 3, 107548, 107260, 107260, 288, 256, 3078389392ULL,
     Reuse::kRam, 34, 5, 200},  // r8
    {"more-ernie-row3427--p05", 3, -1, 130018, 107260, 107260, kUnknownTokens, kUnknownTokens, 3078389392ULL,
     Reuse::kRam, 39, 0, 0},  // r9
}};

static_assert(kDeepRequests.size() == 46);
static_assert(kForkRequests.size() == 31);
// 6267's C0 fork run completed seven requests, then reset the connection on
// row3427 p05 after selecting P=107260. Unknown counters are never passes.
static_assert(kFork6267Requests.size() == 8);
static_assert(kFork6267Requests.back().status == 0);
static_assert(kFork6267Requests.back().observed_prefill == kUnknownTokens);

}  // namespace gufo::tests::component_sequence_trace

#endif  // GUFO_TESTS_CLI_COMPONENT_SEQUENCE_TRACE_HPP_

