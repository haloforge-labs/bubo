// GPU test for the GPU retry of a faulted NPU prefill chunk (L-REL-2). It needs
// a GPU and the model, but no NPU: a chunk that fails part-way (cancelled in
// layer L) leaves the same kind of dirty session state as a faulted NPU chunk.
//
// npu_retry_gpu_test --model FIRST_SHARD.gguf [--chunk 4096] [--context N]
// or GUFO_QFN_MODEL=FIRST_SHARD.gguf. Exit 77 (skipped) without a model.
//
// Arm 1  restore: run A, back up, run a different chunk that fails in layer L,
//        restore, run B, run C. Compare bitwise with a fresh session that ran
//        A, B, C: the session payload after A (restore) and after B, and the
//        logits of B and C.
// Arm 2  chunk grid: the same tokens through other chunk grids on the GPU.
//        Informational: whether a retried chunk can differ from a GPU-only run
//        that cut the prompt elsewhere.
// Arm 3  backup cost: device-event time of the copy, against 0.5 % of a 3.2 s
//        NPU chunk.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

#include "src/core/gguf_reader.hpp"
#include "src/models/qwen/tokenizer.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/device_model.hpp"
#include "src/models/qwen38_flash_next/kernels/rocm/executor.hpp"
#include "src/models/qwen38_flash_next/ngram.hpp"
#include "src/models/qwen38_flash_next/npu_retry.hpp"
#include "src/models/qwen38_flash_next/weights.hpp"

namespace q = gufo::models::qwen38_flash_next;
using Executor = q::rocm::Executor;

namespace {
int failures = 0;

void Check(bool ok, const std::string& what) {
  std::printf("%s %s\n", ok ? "PASS" : "FAIL", what.c_str());
  failures += ok ? 0 : 1;
}

// Which bytes of two session payloads differ, per section of the payload layout
// (Executor::SnapshotSegments: the serializer's own walk).
struct SegmentDiff {
  std::uint64_t bytes = 0;
  std::uint64_t first = ~std::uint64_t{0};  // first differing payload offset
  unsigned segments = 0, total = 0;
  std::vector<unsigned> instances;  // which same-named sections differ
};
using DiffMap = std::map<std::string, SegmentDiff>;

DiffMap Diff(const std::vector<Executor::SnapshotSegment>& segments,
             const std::vector<std::uint8_t>& a,
             const std::vector<std::uint8_t>& b) {
  DiffMap diff;
  std::map<std::string, unsigned> seen;
  for (const auto& s : segments) {
    auto& d = diff[s.name];
    const unsigned instance = seen[s.name]++;
    ++d.total;
    std::uint64_t differing = 0, first = ~std::uint64_t{0};
    for (std::uint64_t i = s.offset; i < s.offset + s.bytes; ++i) {
      const bool outside = i >= a.size() || i >= b.size();
      if (outside || a[i] != b[i]) {
        ++differing;
        first = std::min(first, i);
      }
    }
    if (differing == 0)
      continue;
    d.bytes += differing;
    d.first = std::min(d.first, first);
    ++d.segments;
    d.instances.push_back(instance);
  }
  if (a.size() != b.size())
    diff["payload size"].bytes +=
        a.size() > b.size() ? a.size() - b.size() : b.size() - a.size();
  return diff;
}

std::uint64_t DiffBytes(const DiffMap& diff) {
  std::uint64_t sum = 0;
  for (const auto& [name, d] : diff)
    sum += d.bytes;
  return sum;
}

void PrintDiff(const std::string& what, const DiffMap& diff) {
  if (DiffBytes(diff) == 0) {
    std::printf("DIFF %s: identical\n", what.c_str());
    return;
  }
  for (const auto& [name, d] : diff) {
    if (d.bytes == 0)
      continue;
    std::printf(
        "DIFF %s: %-18s %u of %u sections differ, %llu bytes, first at "
        "payload offset %llu, sections:",
        what.c_str(), name.c_str(), d.segments, d.total,
        static_cast<unsigned long long>(d.bytes),
        static_cast<unsigned long long>(d.first));
    for (std::size_t i = 0; i < d.instances.size() && i < 8; ++i)
      std::printf(" #%u", d.instances[i]);
    std::printf("%s\n", d.instances.size() > 8 ? " ..." : "");
  }
}

// What the backup covers (exactly the accumulative state; the header holds the
// n-gram window), and what the design leaves to the rerun of the chunk.
bool CoveredByBackup(const std::string& name) {
  return name == "header" || name == "conv state" ||
         name == "recurrent state" || name == "PLE history";
}
bool RewrittenByRerun(const std::string& name) {
  return name == "K cache" || name == "V cache" || name == "indexer tail" ||
         name == "indexer wrap" || name == "pooled block keys";
}
std::uint64_t BytesWhere(const DiffMap& diff,
                         bool (*pick)(const std::string&)) {
  std::uint64_t sum = 0;
  for (const auto& [name, d] : diff)
    if (pick(name))
      sum += d.bytes;
  return sum;
}
bool Unclassified(const std::string& name) {
  return !CoveredByBackup(name) && !RewrittenByRerun(name);
}

bool SameBits(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() &&
         std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

struct Rig {
  std::unique_ptr<gufo::core::GgufReader> reader;
  std::optional<q::ModelWeights> weights;
  std::unique_ptr<gufo::tokenization::QwenTokenizer> tokenizer;
  std::unique_ptr<q::NgramTable> ngram;
  std::unique_ptr<q::rocm::DeviceModel> device;
  std::unique_ptr<Executor> executor;
  std::uint32_t context = 0;
};

// One session and the logits row of its last chunk.
struct Run {
  Rig& rig;
  std::unique_ptr<q::rocm::Session> session;
  std::vector<float> logits;
  explicit Run(Rig& r) : rig(r) {
    std::string error;
    session = rig.executor->CreateSession(
        gufo::core::SessionMode::kAutoregressive, rig.context, &error);
    if (!session) {
      std::fprintf(stderr, "session failed: %s\n", error.c_str());
      std::exit(1);
    }
    logits.resize(rig.weights->config.vocab_size);
  }
  bool Chunk(std::span<const std::int32_t> tokens,
             std::span<const std::int32_t> next, std::string* error) {
    return rig.executor->Forward(*session, tokens, 1, logits.data(),
                                 Executor::ForwardMode::kPrefill, error,
                                 nullptr, false, next);
  }
  std::vector<std::uint8_t> Payload() {
    std::string error;
    std::vector<std::uint8_t> bytes(rig.executor->SnapshotBytes(*session, 0));
    if (!rig.executor->SaveSnapshot(*session, 0, bytes, &error)) {
      std::fprintf(stderr, "snapshot failed: %s\n", error.c_str());
      std::exit(1);
    }
    return bytes;
  }
};

std::vector<std::int32_t> MakeTokens(Rig& rig, std::size_t count) {
  std::vector<std::int32_t> tokens;
  for (unsigned i = 0; tokens.size() < count; ++i) {
    const std::string line =
        "Entry " + std::to_string(i) + ": the lamp number " +
        std::to_string(i * 7919 % 1013) + " in room " + std::to_string(i % 37) +
        " was checked on day " + std::to_string(1 + i % 28) + "; reading " +
        std::to_string(i * 31 % 977) + " units, status " +
        (i % 3 ? "steady" : "flickering") + ".\n";
    for (auto id : rig.tokenizer->Encode(line))
      tokens.push_back(static_cast<std::int32_t>(id));
  }
  tokens.resize(count);
  return tokens;
}

// Feeds tokens in chunks of `grid` and returns the last logits row.
std::vector<float> Feed(Rig& rig, std::span<const std::int32_t> tokens,
                        std::size_t grid) {
  Run run(rig);
  for (std::size_t off = 0; off < tokens.size(); off += grid) {
    const auto n = std::min(grid, tokens.size() - off);
    const auto after = std::min(grid, tokens.size() - off - n);
    std::string error;
    if (!run.Chunk(tokens.subspan(off, n), tokens.subspan(off + n, after),
                   &error)) {
      std::fprintf(stderr, "prefill failed: %s\n", error.c_str());
      std::exit(1);
    }
  }
  return run.logits;
}

void RestoreArm(Rig& rig, std::span<const std::int32_t> tokens,
                std::size_t chunk) {
  const auto a = tokens.subspan(0, chunk);
  const auto b = tokens.subspan(chunk, chunk);
  const auto c = tokens.subspan(2 * chunk, chunk);
  std::vector<std::int32_t> dirty(b.rbegin(), b.rend());  // other tokens
  std::string error;

  Run fresh(rig);
  Check(fresh.Chunk(a, b, &error), "fresh: chunk A");
  const auto fresh_after_a = fresh.Payload();
  Check(fresh.Chunk(b, c, &error), "fresh: chunk B");
  const auto fresh_b = fresh.logits;
  const auto fresh_after_b = fresh.Payload();
  Check(fresh.Chunk(c, {}, &error), "fresh: chunk C");
  const auto fresh_c = fresh.logits;

  // The layers whose start cancels the chunk: early, middle, late. The
  // cancellation check runs once before the layers and once per layer.
  for (const std::uint32_t layer : {1u, 11u, 30u, 47u}) {
    const std::string tag = "cancel in layer " + std::to_string(layer) + ": ";
    Run run(rig);
    Check(run.Chunk(a, dirty, &error), tag + "chunk A");
    Check(rig.executor->SaveNpuRetryBackup(*run.session, &error),
          tag + "backup");
    double copy_ms = -1;
    const auto payload_after_a = run.Payload();
    Check(payload_after_a == fresh_after_a,
          tag + "payload after A equals fresh");

    std::uint32_t polls = 0;
    run.session->SetCancellationCheck(
        [&polls, layer] { return ++polls > layer + 1; });
    const bool failed = !run.Chunk(dirty, {}, &error);
    run.session->SetCancellationCheck({});
    Check(failed, tag + "dirtying chunk failed");
    Check(run.session->position() == chunk, tag + "position not advanced");
    const auto dirty_payload = run.Payload();
    Check(dirty_payload != payload_after_a,
          tag + "the failed chunk changed the session state");
    (void)rig.executor->NpuRetryBackupMs(&copy_ms);

    Check(rig.executor->RestoreNpuRetryBackup(*run.session, &error),
          tag + "restore");
    const auto segments = rig.executor->SnapshotSegments(*run.session, 0);
    // The snapshot copies on its own stream and expects the session's work to
    // be complete ("callers have completed that session's model operation");
    // the restore copies are only queued on the compute stream. Read once
    // without waiting (diagnostic only) and once after the stream has drained.
    const auto payload_unsynced = run.Payload();
    PrintDiff(tag + "after restore, read before the stream drained",
              Diff(segments, payload_after_a, payload_unsynced));
    (void)hipStreamSynchronize(rig.executor->stream());
    const auto payload_restored = run.Payload();
    const auto restored = Diff(segments, payload_after_a, payload_restored);
    PrintDiff(tag + "after restore, stream drained", restored);
    // (a) exactly what the backup covers is back, bit for bit.
    Check(BytesWhere(restored, CoveredByBackup) == 0,
          tag +
              "backup-covered sections (header with the n-gram window, conv "
              "state, recurrent state, PLE history) equal payload after A "
              "after restore (bitwise)");
    // (b) anything else that differs must be a section the design leaves to
    // the rerun of the chunk (K/V rows, indexer rows, pooled block keys); the
    // check "payload after B equals fresh" below proves the rerun wrote them.
    Check(BytesWhere(restored, Unclassified) == 0,
          tag + "differing bytes are only in sections the rerun rewrites");
    std::printf(
        "INFO %swhole payload after restore %s payload after A\n", tag.c_str(),
        payload_restored == payload_after_a ? "equals" : "differs from");
    Check(run.Chunk(b, c, &error), tag + "retry chunk B");
    Check(SameBits(run.logits, fresh_b), tag + "logits of B equal fresh");
    Check(run.Payload() == fresh_after_b, tag + "payload after B equals fresh");
    Check(run.Chunk(c, {}, &error), tag + "chunk C");
    Check(SameBits(run.logits, fresh_c), tag + "logits of C equal fresh");
  }
}

void GridArm(Rig& rig, std::span<const std::int32_t> tokens,
             std::size_t chunk) {
  const auto reference = Feed(rig, tokens, chunk);
  for (const std::size_t grid : {chunk / 2, chunk / 4, chunk + 128}) {
    if (grid == 0 || grid > rig.executor->max_batch())
      continue;
    const bool same = SameBits(Feed(rig, tokens, grid), reference);
    std::printf("INFO grid %zu vs %zu on %zu tokens: logits %s\n", grid, chunk,
                tokens.size(), same ? "bit-identical" : "DIFFER");
  }
}







struct CostStats {
  double median = 0, p95 = 0, max = 0;
};
CostStats Summarize(std::vector<double> ms) {
  std::sort(ms.begin(), ms.end());
  return {ms[ms.size() / 2], ms[ms.size() * 95 / 100], ms.back()};
}

void CostArm(Rig& rig, std::span<const std::int32_t> tokens,
             std::size_t chunk) {
  Run run(rig);
  std::string error;
  Check(run.Chunk(tokens.subspan(0, chunk), {}, &error), "cost: chunk");
  constexpr std::size_t kJunk = std::size_t{1} << 30;  // 1 GiB of traffic
  void* junk = nullptr;
  hipStream_t other = nullptr;
  const bool have_traffic =
      hipMalloc(&junk, kJunk) == hipSuccess &&
      hipStreamCreateWithFlags(&other, hipStreamNonBlocking) == hipSuccess;
  Check(have_traffic, "cost: traffic buffer and stream");
  const auto stream = rig.executor->stream();
  struct Condition {
    const char* name;
    std::function<void()> before;
  };
  std::vector<Condition> conditions = {
      {"hot, back to back", [] {}},
      {"after 300 ms idle",
       [] { std::this_thread::sleep_for(std::chrono::milliseconds(300)); }},
      {"behind GPU work on the stream",
       [&] { (void)hipMemsetAsync(junk, 0, kJunk, stream); }},
      {"with 1 GiB of traffic on another stream",
       [&] { (void)hipMemsetAsync(junk, 1, kJunk, other); }}};
  for (std::size_t k = 0; k < conditions.size(); ++k) {
    const auto& condition = conditions[k];
    if (!have_traffic && k >= 2)
      continue;
    std::vector<double> ms;
    for (int i = 0; i < 30; ++i) {
      condition.before();
      Check(rig.executor->SaveNpuRetryBackup(*run.session, &error),
            "cost: save");
      (void)hipStreamSynchronize(stream);
      if (other)
        (void)hipStreamSynchronize(other);
      double copy = -1;
      Check(rig.executor->NpuRetryBackupMs(&copy) && copy > 0,
            "cost: event time");
      ms.push_back(copy);
    }
    const auto stats = Summarize(ms);
    std::printf(
        "INFO backup (%s): median_ms=%.3f p95_ms=%.3f max_ms=%.3f "
        "(limit 16 ms = 0.5 %% of a 3.2 s NPU chunk)\n",
        condition.name, stats.median, stats.p95, stats.max);
    Check(stats.p95 <= 16.0,
          std::string("cost: p95 within 0.5 % of a 3.2 s chunk, ") +
              condition.name);
  }
  const auto bytes = rig.executor->NpuRetryBackupAllocatedBytes();
  std::printf(
      "INFO backup: bytes=%zu in %zu launches\n", bytes,
      q::NpuRetryLaunches(q::NpuRetryRegions(rig.weights->config).size()));
  Check(bytes == q::NpuRetryBackupBytes(rig.weights->config, true),
        "cost: allocated bytes equal NpuRetryBackupBytes");
  if (other)
    (void)hipStreamDestroy(other);
  if (junk)
    (void)hipFree(junk);
}
}  // namespace

int main(int argc, char** argv) {
  std::string model_path;
  std::uint32_t chunk = 4096, context = 0;
  if (const char* env = std::getenv("GUFO_QFN_MODEL"))
    model_path = env;
  for (int i = 1; i + 1 < argc; i += 2) {
    const std::string arg = argv[i];
    if (arg == "--model")
      model_path = argv[i + 1];
    else if (arg == "--chunk")
      chunk =
          static_cast<std::uint32_t>(std::strtoul(argv[i + 1], nullptr, 10));
    else if (arg == "--context")
      context =
          static_cast<std::uint32_t>(std::strtoul(argv[i + 1], nullptr, 10));
  }
  if (model_path.empty()) {
    std::printf("SKIP: no model (--model or GUFO_QFN_MODEL)\n");
    return 77;
  }
  if (chunk < 256 || chunk > 4096) {
    std::fprintf(stderr, "--chunk is 256..4096\n");
    return 2;
  }
  if (context == 0)
    context = 3 * chunk + 512;

  Rig rig;
  rig.context = context;
  std::string error;
  rig.reader = gufo::core::GgufReader::OpenFile(model_path, &error);
  if (!rig.reader) {
    std::fprintf(stderr, "open failed: %s\n", error.c_str());
    return 1;
  }
  rig.weights = q::ModelWeights::Bind(*rig.reader, &error);
  if (!rig.weights) {
    std::fprintf(stderr, "bind failed: %s\n", error.c_str());
    return 1;
  }
  rig.tokenizer =
      gufo::tokenization::QwenTokenizer::CreateFromGguf(*rig.reader, &error);
  if (!rig.tokenizer) {
    std::fprintf(stderr, "tokenizer failed: %s\n", error.c_str());
    return 1;
  }
  const auto& c = rig.weights->config;
  if (c.ple_layer >= 0) {
    const auto& t = rig.weights->ple_table;
    rig.ngram = q::NgramTable::Open(
        rig.reader->GetMappedRegions()[t.shard].file_descriptor, t.file_offset,
        t.rows, c.ple_head_dim, t.type, &error);
    if (!rig.ngram) {
      std::fprintf(stderr, "n-gram table failed: %s\n", error.c_str());
      return 1;
    }
  }
  rig.device = q::rocm::DeviceModel::Upload(*rig.weights, *rig.reader, nullptr,
                                            nullptr, &error);
  if (!rig.device) {
    std::fprintf(stderr, "upload failed: %s\n", error.c_str());
    return 1;
  }
  Executor::Options options;
  options.max_batch = chunk + 128;
  options.max_logit_rows = 1;
  rig.executor =
      Executor::Create(*rig.device, rig.ngram.get(), options, &error);
  if (!rig.executor) {
    std::fprintf(stderr, "executor failed: %s\n", error.c_str());
    return 1;
  }

  const auto tokens = MakeTokens(rig, 3 * std::size_t{chunk});
  std::printf("INFO %zu tokens, chunk %u, backup bytes %zu\n", tokens.size(),
              chunk, q::NpuRetryBackupBytes(c, true));
  RestoreArm(rig, tokens, chunk);
  GridArm(rig, tokens, chunk);
  CostArm(rig, tokens, chunk);
  std::printf("%s (%d failed checks)\n", failures ? "FAIL" : "PASS", failures);
  return failures ? 1 : 0;
}
