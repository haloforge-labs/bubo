# Lean host-flow adaptation (2026-10-12)

The lean adapter/session additionally adapt the MIT fork's Process buffer flow:
resident weights finalized at load, one A sync per layer call, up to four
in-flight runs, completion before C readback, and 1000 ms / 2000 ms keepalive.
Source remains mikealanni/gufo 8e0fb1dedbc06c2b9d2ed6cc429cc394b840dee7;
Copyright (c) 2026 gufo contributors, root LICENSE (MIT).
The old P2/P3 operational descriptions below are historical provenance.
The engine no longer links durable tracing, starts a watchdog, checks launcher
evidence, allocates canaries, scans unused BOs, or holds the host on failure.
On uncertain DMA, the adapter retains its runs/pages until process exit like
the fork, without clearing active run ownership. Power mode is read-only.

# NPU adapter provenance and notices

P10 supplies private qualification staging, an operator runbook, an OFF
equivalence protocol and an unarmed device-run record under `tools/npu`.
Staging preserves the complete repository/fork MIT license, the complete
mlir-aie Apache-2.0 WITH LLVM-exception license, and XRT Apache-2.0 LICENSE
and NOTICE. It copies no AIE source, xclbin/instructions, vendor DLL, driver
or SDK/header tree. The precompiled pair remains hashed external input;
artifact redistribution rights are NOT established by its source SPDX
notice or by this packet. Public/standalone packaging remains deferred.
No device qualification or public support promise follows from staging.

P6 selectively adapts BFP group encoding/decoding and A/B/C tile geometry
from `kernels.hip.cpp` at the same pinned fork commit. Root MIT terms and
Copyright (c) 2026 gufo contributors apply. No AIE source or vendor runtime
was copied; the existing AIE Apache-2.0 WITH LLVM-exception notices below
remain unchanged. See BFP.md for the CPU contract and unvalidated HIP scope.

P5 selectively adapts only Q8_0 lane geometry from the same fork's
`src/models/qwen38_flash_next/kernels/rocm/device_model.cpp`. Its MIT notice
below applies; no AIE source or new vendor runtime is copied by P5.

`private_bo_adapter.cpp` selectively extracts and adapts `Engine` from the
completed `npu-tooling-sol/host.cpp`. `tests/npu/fake_xrt.cpp` extracts that
job's fake implementations against the same pinned upstream class declarations.
These are the single classic-XRT layer used by this port. No replacement class
layouts or fork-wide performance changes are included. Repository MIT terms
apply, including Copyright (c) 2026 gufo contributors (see root LICENSE).

The readonly fork is mikealanni/gufo at
`8e0fb1dedbc06c2b9d2ed6cc429cc394b840dee7`. Its root MIT notice is preserved
by the same root LICENSE. P3 selectively adapts the fork's 1000 ms keepalive
and 2000 ms quiet semantics to a single serialized lifetime worker, following
the newer Workaround-B addendum. No fork abort, userptr, HIP allocation or
POSIX trace implementation was copied.

The fixed shape, instruction/artifact hashes, memory connectivity and unused
slot policy come from the tooling static proof: `slot-contract.txt`,
`artifact-static.json`, `transaction-static.json` and `transaction-evidence.txt`.
The 608 operations contain 160 DDR patches referring only to A/B/C. The maximum
contiguous ends are 11,796,480 / 2,949,120 / 4,718,592 bytes. XML slots 6 and 7
connect to HOST group 0 and are explicitly bound to zeroed A-sized quarantine
BOs with 4096-byte canaries. Static proof is not hardware acceptance.

The associated AIE sources `tools/npu/mm_bfp_mixed.cc` and
`tools/npu/n32_core_placed.py` retain their original notices:

> Copyright (C) 2026, Advanced Micro Devices, Inc.
> SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Those sources and precompiled artifacts remain readonly external inputs here;
they are not redistributed by this packet. Preserve their full Apache-2.0 WITH
LLVM-exception notices if copied by a later packet.

XRT public headers are referenced, unmodified, from the tooling checkout at
`15e6319be8de1e76a6150111a3861729e988fdb5`. Their Apache-2.0 notices and complete
LICENSE remain in that checkout. The existing CMake layer snapshots only the
tooling-generated import library per configuration. No vendor DLL is copied.

P3 supplies a durable EventSink: every CallOutcome is merged with the same
run sequencer's seq/utc/qpc/qpc_freq/phase/stage_id, then appended and flushed to
npu-trace.jsonl. P2 records every call begin/outcome, dispatch_seq, ert_cmd_state,
native error code (null if unavailable), full error text and possible-DMA state.
P4 must authorize immutable effective ON before calling Create. No production
call site or runtime environment enabler is introduced by P2/P3.

P3 lifecycle contract: call NpuSession::Start once, after immutable ON
authorization and before GPU warm-up. All six private BOs/device/context/kernel
stay registered for the server lifetime. A separate CPU-only watchdog must
flush readiness before Create. Timeout defaults to 8000 ms (strict 1..8000);
GUFO_NPU_KEEPALIVE is strict 0|1, GUFO_NPU_KEEPALIVE_MS is 0 or 100..60000
(default 1000), and GUFO_NPU_TRACE is strict 0|1. Interval 0 / KEEPALIVE=0
disables periodic heartbeats, not the bounded startup exposure. Pause via
YieldKeepalive(true) is a completion barrier; resume is explicit. All heartbeat
dispatches count as exposure and carry kind=heartbeat. There is no HIP callback
acceptance-as-completion shortcut, front/pop lifetime alias, or NewBuffer race.

Every trace record uses checked WriteFile + FlushFileBuffers and write-through;
ring.bin complements it with the last 4096 driver records. The run directory
defaults to <local path> The independent watcher
flushes heartbeat.csv every 250 ms, stops new work on three missing 1 s beats,
and detects any active XRT/cleanup call or lost worker progress at 30 s. It
never kills/restarts a host. The caller's phase barrier is 30.5 s plus Logger's
bounded flush, queue locks/event waits are 100 ms, and Shutdown/destruction
wait at most 2 s each. A stalled worker retains shared ownership; uncertain
DMA parks it for process lifetime. owner_recovery=true is a REQUIRED host
recovery hold for P4/P9 integration, not permission to exit the host. No device
qualification or prevention of an OS/fabric hang follows from these deadlines.

Power mode is queried read-only through the same adapter's public XRT platform
JSON. Only verified Performance permits the initial GEMM; Turbo, unsupported,
missing or erroneous mode refuses startup without compute. P3 adds no setter:
the pinned source lacks the Windows Ryzen backend needed to prove its admin
policy/persistence. P10 must document the owner-controlled Performance preflight
and before/after record, driver-reload recheck, and FLM dictation/context-limit
coexistence test, including the keepalive yield mitigation. Numerical correctness,
actual power-state residency and device teardown remain unvalidated.

P7 selectively adapts the linear-attention z lane split from the same MIT
`mikealanni/gufo` fork at `8e0fb1de`. `linear_z.cpp`, `linear_z_hip.cpp`, the
qualified model staging and the bounded GPU-prefix launch retain MIT SPDX
notices. The Windows private-copy/lifetime orchestration and CPU/fake tests
are new port code. No unrelated fork GPU optimization or AIE source was
transplanted. The existing MIT and AIE Apache-2.0 WITH LLVM-exception notices
above continue to apply to their respective sources/artifacts.

P8 selectively adapts the attention q/gate head split and shared-expert
lane selection from the same MIT fork at `8e0fb1de`. `projection.cpp`, the
owned HIP bridge and bounded GPU suffix are new Windows private-copy
orchestration using P2/P3's single XRT adapter. MIT SPDX notices remain;
no additional AIE source, wrapper or vendor runtime was copied. Existing
AIE Apache-2.0 WITH LLVM-exception notices remain applicable and unchanged.
