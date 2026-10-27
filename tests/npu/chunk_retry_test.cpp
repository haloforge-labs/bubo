// SPDX-License-Identifier: MIT
// CPU only: the pure parts of the GPU retry of a faulted NPU chunk (L-REL-2).
#include "src/npu/chunk_retry.hpp"

#include <cassert>
#include <cstdint>
#include <iostream>
#include <optional>
#include <string>

#include "src/models/qwen38_flash_next/npu_retry.hpp"
#include "src/npu/fault_state.hpp"
#include "src/npu/fault_telemetry.hpp"
#include "src/npu/runtime_status.hpp"

using namespace gufo::npu;
namespace qfn = gufo::models::qwen38_flash_next;

namespace {
bool Has(const std::string& text, const std::string& part) {
  return text.find(part) != std::string::npos;
}

void Decision() {
  // Retry iff the flag flipped during this chunk, the chunk used the NPU and
  // this is the first attempt.
  assert(ShouldRetryChunk(false, true, true, true));
  assert(!ShouldRetryChunk(true, true, true, true));    // already failed
  assert(!ShouldRetryChunk(false, false, true, true));  // no fault
  assert(!ShouldRetryChunk(true, false, true, true));
  assert(!ShouldRetryChunk(false, true, false, true));  // chunk ran on the GPU
  assert(!ShouldRetryChunk(false, true, true, false));  // one attempt only
  for (int bits = 0; bits < 16; ++bits) {
    const bool before = bits & 1, after = bits & 2, used = bits & 4,
               first = bits & 8;
    assert(ShouldRetryChunk(before, after, used, first) ==
           (!before && after && used && first));
  }
}

void BackupDecision() {
  // lanes ready, the pass qualifies for a lane, no hold that stays on.
  assert(ShouldBackupChunk(true, true, false));
  assert(!ShouldBackupChunk(false, true, false));  // NPU not ready
  assert(!ShouldBackupChunk(true, false, false));  // n below the minimum
  assert(!ShouldBackupChunk(true, true, true));    // hold stays on
  // No backup means no retry: a chunk that cannot use a lane (a tail below
  // 4096 tokens) never reaches ShouldRetryChunk with a backup in hand.
  bool backed_up = ShouldBackupChunk(true, false, false);
  assert(!(backed_up && ShouldRetryChunk(false, true, true, true)));
  // ChunkUsed() is the previous chunk's value if the pass failed before its
  // chunk start: only a moved serial counts.
  assert(ChunkUsedThisAttempt(7, 8, true));
  assert(!ChunkUsedThisAttempt(7, 7, true));
  assert(!ChunkUsedThisAttempt(7, 8, false));
  // The backup time belongs to the chunk noted during this attempt only.
  assert(BackupBelongsToChunk(10, 11));
  assert(!BackupBelongsToChunk(10, 10));  // failed before the check
  assert(!BackupBelongsToChunk(10, 12));
}

void LogLine() {
  assert(FormatChunkRetryWarning(2, 1.234, 3101.26) ==
         "chunk 2 faulted; retried on GPU (copy_ms=1.23 retry_ms=3101.3)");
  assert(FormatChunkRetryWarning(1, -1, 0.04) ==
         "chunk 1 faulted; retried on GPU (copy_ms=na retry_ms=0.0)");
}

qfn::Config Production() {
  qfn::Config c;
  c.num_layers = 48;
  c.full_attention_interval = 4;
  c.hidden_size = 2560;
  c.hc_count = 4;
  c.ssm_conv_kernel = 4;
  c.ssm_head_dim = 128;
  c.ssm_num_k_heads = 16;
  c.ssm_num_v_heads = 48;
  c.ple_layer = 1;
  c.ple_conv_kernel = 4;
  c.ple_ngram_size = 3;
  return c;
}

void BackupBytes() {
  auto c = Production();
  const auto layout = qfn::NpuRetryLayoutFor(c);
  assert(layout.linear_layers == 36);
  assert(layout.conv_floats == 3 * (2 * 16 * 128 + 48 * 128));  // 30720
  assert(layout.state_floats == 48 * 128 * 128);                // 786432
  assert(layout.ple_floats == 9 * 10240);                       // 92160
  // 36 * (30720 + 786432) * 4 + 9 * 10240 * 4
  assert(qfn::NpuRetryBackupBytes(c, true) == 117669888 + 368640);
  assert(qfn::NpuRetryBackupBytes(c, true) == 118038528);
  std::cout << "BACKUP-BYTES production=" << qfn::NpuRetryBackupBytes(c, true)
            << "\n";
  // Disabled NPU prefill: nothing is allocated.
  assert(qfn::NpuRetryBackupBytes(c, false) == 0);
  // No PLE layer: the PLE term drops out.
  c.ple_layer = -1;
  assert(qfn::NpuRetryBackupBytes(c, true) == 117669888);
  assert(qfn::NpuRetryBackupBytes(c, false) == 0);
}

struct FakeDevice {
  int allocs = 0, frees = 0, events = 0, destroyed = 0, cleared = 0;
  int fail_event_at = 0;  // 1-based create_event call that fails (0 = none)
  bool fail_alloc = false;
  int created = 0;
  int storage[2] = {0, 0};
  qfn::NpuRetryBackupOps Ops() {
    qfn::NpuRetryBackupOps ops;
    ops.alloc = [this](std::size_t) -> void* {
      if (fail_alloc)
        return nullptr;
      ++allocs;
      return &storage[0];
    };
    ops.free = [this](void*) { ++frees; };
    ops.create_event = [this]() -> void* {
      if (++created == fail_event_at)
        return nullptr;
      ++events;
      return &storage[1];
    };
    ops.destroy_event = [this](void*) { ++destroyed; };
    ops.clear_error = [this] { ++cleared; };
    return ops;
  }
};

void BackupResources() {
  {  // success: everything exists, nothing cleared, Release gives it all back
    FakeDevice d;
    qfn::NpuRetryBackupResources r(d.Ops());
    assert(!r.allocated() && r.bytes() == 0);
    assert(r.Ensure(1000) && r.allocated() && r.bytes() == 1000);
    assert(r.Ensure(1000) && d.allocs == 1 && d.events == 2);  // no second one
    assert(d.cleared == 0 && !r.failed());
    r.Release();
    assert(!r.allocated() && r.bytes() == 0 && d.frees == 1 &&
           d.destroyed == 2);
    r.Release();  // idempotent
    assert(d.frees == 1 && d.destroyed == 2);
    assert(r.Ensure(1000) && d.allocs == 2);  // a release is not a failure
  }
  // Each failure path: what was half-created is given back, the sticky error
  // is cleared once, the failure is remembered and nothing is tried again.
  for (int variant = 0; variant < 3; ++variant) {
    FakeDevice d;
    d.fail_event_at = variant == 0 ? 1 : variant == 1 ? 2 : 0;
    d.fail_alloc = variant == 2;
    qfn::NpuRetryBackupResources r(d.Ops());
    assert(!r.Ensure(1000) && r.failed() && !r.allocated());
    assert(d.cleared == 1);
    assert(d.events == d.destroyed && d.allocs == d.frees);  // no leak
    const int created = d.created, allocs = d.allocs;
    d.fail_event_at = 0;
    d.fail_alloc = false;
    assert(!r.Ensure(1000));  // the first failure sticks: retry stays off
    assert(d.created == created && d.allocs == allocs && d.cleared == 1);
  }
  {  // destruction releases
    FakeDevice d;
    {
      qfn::NpuRetryBackupResources r(d.Ops());
      assert(r.Ensure(8));
    }
    assert(d.frees == 1 && d.destroyed == 2);
  }
}

void BackupRegions() {
  // The production model: conv + recurrent state of 36 linear layers, then PLE.
  auto c = Production();
  const auto regions = qfn::NpuRetryRegions(c);
  assert(regions.size() == 73);
  std::size_t sum = 0, expected_offset = 0;
  for (std::size_t i = 0; i < regions.size(); ++i) {
    const auto& r = regions[i];
    assert(r.offset == expected_offset);   // back to back, in buffer order
    assert(r.count > 0 && r.Aligned16());  // every region starts on 16 bytes
    expected_offset += r.count;
    sum += r.count;
    if (i < 72) {
      assert(r.kind ==
             (i % 2 ? qfn::NpuRetryKind::kState : qfn::NpuRetryKind::kConv));
      assert(c.IsLinearLayer(r.layer));
      assert(r.layer == regions[i - i % 2].layer);
      assert(r.count == (i % 2 ? 786432U : 30720U));
    }
  }
  assert(regions[72].kind == qfn::NpuRetryKind::kPle &&
         regions[72].count == 92160);
  assert(regions[0].layer == 0 && regions[70].layer == 46 &&
         regions[71].layer == 46);
  assert(sum == 36 * (30720 + 786432) + 92160);
  assert(sum * sizeof(float) ==
         qfn::NpuRetryBackupBytes(c, true));  // 118,038,528
  assert(sum * sizeof(float) == 118038528);
  // 73 regions: two launches of at most 64; the table is a kernel argument
  // (24 bytes a region), so no device table is allocated.
  assert(qfn::NpuRetryLaunches(regions.size()) == 2 && qfn::NpuRetryFits(73));
  assert(qfn::NpuRetryLaunches(0) == 0 && qfn::NpuRetryLaunches(1) == 1 &&
         qfn::NpuRetryLaunches(64) == 1 && qfn::NpuRetryLaunches(65) == 2 &&
         qfn::NpuRetryLaunches(256) == 4 && qfn::NpuRetryLaunches(257) == 5);
  assert(qfn::NpuRetryFits(256) && !qfn::NpuRetryFits(257));
  std::cout << "BACKUP-REGIONS production=" << regions.size() << " in "
            << qfn::NpuRetryLaunches(regions.size()) << " launches, " << 24 * 64
            << " argument bytes a launch\n";
  // Without PLE the last region is gone.
  c.ple_layer = -1;
  assert(qfn::NpuRetryRegions(c).size() == 72);
  // Edge: counts that are not multiples of 4 floats leave later regions
  // unaligned for a 16-byte copy (the kernel then copies them element by
  // element); the sum still equals the buffer size and nothing overlaps.
  qfn::Config odd = Production();
  odd.ssm_conv_kernel = 2;   // conv = 1 * channels = 10240 (aligned)
  odd.ssm_num_v_heads = 47;  // state = 47 * 128 * 128 (aligned)
  odd.hc_count = 1;
  odd.hidden_size = 2561;  // PLE history = 9 * 2561 = 23049 floats (odd)
  const auto odd_regions = qfn::NpuRetryRegions(odd);
  std::size_t odd_sum = 0;
  for (std::size_t i = 0; i < odd_regions.size(); ++i) {
    assert(odd_regions[i].offset == odd_sum);
    odd_sum += odd_regions[i].count;
  }
  assert(odd_sum * sizeof(float) == qfn::NpuRetryBackupBytes(odd, true));
  assert(odd_regions.back().kind == qfn::NpuRetryKind::kPle &&
         odd_regions.back().count == 23049 && odd_regions.back().Aligned16());
  // Make a region start off the 16-byte boundary: an odd conv count.
  odd.ple_layer = -1;
  odd.ssm_num_k_heads = 1;
  odd.ssm_head_dim = 3;
  odd.ssm_num_v_heads = 1;  // channels = 2*3 + 3 = 9, conv = 9, state = 9
  const auto tiny = qfn::NpuRetryRegions(odd);
  bool saw_unaligned = false;
  for (const auto& r : tiny)
    saw_unaligned = saw_unaligned || !r.Aligned16();
  assert(saw_unaligned && tiny.front().Aligned16());
  // No linear layer: no region, no bytes.
  qfn::Config none = Production();
  none.full_attention_interval = 1;  // every layer is attention
  none.ple_layer = -1;
  assert(qfn::NpuRetryRegions(none).empty() &&
         qfn::NpuRetryBackupBytes(none, true) == 0);
}

void Switch() {
  std::uint64_t chunk = 99;
  std::string error;
  assert(ParseTestFailChunk("0", &chunk, &error) && chunk == 0);
  assert(ParseTestFailChunk("1", &chunk, &error) && chunk == 1);
  assert(ParseTestFailChunk("1000000", &chunk, &error) && chunk == 1000000);
  for (const char* bad : {"", "-1", "+1", " 1", "1 ", "1.5", "abc", "1x",
                          "1000001", "99999999999999999999"}) {
    chunk = 7;
    error.clear();
    assert(!ParseTestFailChunk(bad, &chunk, &error) && chunk == 7 &&
           !error.empty());
  }
  // Chunk k is counted from 0; the chunk counter of a request from 1.
  assert(!TestFailChunkHits(std::nullopt, 1));
  assert(TestFailChunkHits(0, 1) && !TestFailChunkHits(0, 2));
  assert(TestFailChunkHits(1, 2) && !TestFailChunkHits(1, 1) &&
         !TestFailChunkHits(1, 3));
  // Debug hooks off: the switch is dropped with a short note. On: no note.
  const auto off = TestFailChunkIgnoredNote(false);
  assert(off && Has(*off, "GUFO_NPU_TEST_FAIL_CHUNK ignored") &&
         Has(*off, "debug hooks") && off->size() < 80);
  assert(!TestFailChunkIgnoredNote(true));
  // With hooks it is loud and says which chunk.
  const auto on = TestFailChunkActiveText(1);
  assert(Has(on, "GUFO_NPU_TEST_FAIL_CHUNK override active") &&
         Has(on, "internal test setting, not for normal use") &&
         Has(on, "NPU chunk 1"));
}

void CounterAndStatus() {
  ResetNpuRuntimeStatusForTest();
  assert(CurrentNpuRuntimeStatus().gpu_retries == 0);
  Faults().gpu_retries.fetch_add(1);
  assert(CurrentNpuRuntimeStatus().gpu_retries == 1);
  // The retry count is internal: the contract schema has no key for it.
  const auto json = ToJson(CurrentNpuRuntimeStatus()).dump();
  assert(!Has(json, "retries"));
  ResetNpuRuntimeStatusForTest();
  assert(CurrentNpuRuntimeStatus().gpu_retries == 0);
}

void RequestLine() {
  ResetNpuRuntimeStatusForTest();
  // A backup time before any chunk of the request is not recorded.
  NoteChunkBackup(9.0);
  assert(!Has(TakeRequestTelemetrySuffix(), "chunk_backup_ms"));
  ChunkStart start;
  ChunkEnd end;
  end.duration_ms = 3200.04;
  NoteRequestChunk(start, end);
  NoteChunkBackup(1.5);
  NoteRequestChunk(start, end);
  NoteChunkBackup(2.5);
  NoteRequestChunk(start, end);
  NoteChunkBackup(-1);  // not measured
  Faults().gpu_retries.fetch_add(1);
  const auto line = TakeRequestTelemetrySuffix();
  std::cout << "REQUEST-SUFFIX: " << line << "\n";
  assert(Has(line,
             " chunk_ms=[3200.0,3200.0,3200.0] "
             "chunk_backup_ms=[1.50,2.50,na] backup_ms_total=4.00 "
             "backup_ms_max=2.50"));
  assert(Has(line, " npu_chunk_gpu_retries=1"));
  assert(Has(line, " hal_faults=") && Has(line, " chip_power="));
  // A pass that fails before its check notes no chunk: its backup time must not
  // land in the slot of the chunk before. Mirrors Session::Feed.
  const auto attempt = [](bool check_runs, double ms) {
    const auto before = RequestChunksNoted();
    if (check_runs)
      NoteRequestChunk(ChunkStart{}, ChunkEnd{});
    if (BackupBelongsToChunk(before, RequestChunksNoted()))
      NoteChunkBackup(ms);
  };
  attempt(true, 1.5);
  attempt(false, 9.0);  // failed before the check
  attempt(true, 2.5);
  const auto tidy = TakeRequestTelemetrySuffix();
  assert(Has(tidy,
             " chunk_backup_ms=[1.50,2.50] backup_ms_total=4.00 "
             "backup_ms_max=2.50"));
  assert(!Has(tidy, "9.00"));
  // The next request starts clean; the retry total is a process total.
  NoteRequestChunk(start, end);
  const auto next = TakeRequestTelemetrySuffix();
  // A chunk without a backup time shows na (the live check fails the rule).
  assert(Has(next, " chunk_backup_ms=[na] backup_ms_total=0.00") &&
         Has(next, "npu_chunk_gpu_retries=1"));
  ResetNpuRuntimeStatusForTest();
}
}  // namespace

int main() {
  Decision();
  BackupDecision();
  BackupResources();
  LogLine();
  BackupBytes();
  BackupRegions();
  Switch();
  CounterAndStatus();
  RequestLine();
  std::cout << "PASS GPU retry of a faulted NPU chunk: decision, log line, "
               "backup bytes, switch, counter, request line\n";
}
