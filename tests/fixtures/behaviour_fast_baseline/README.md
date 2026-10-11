# Frozen sampler oracle

`src/core/{sampling.cpp,sampling.hpp,behaviour.hpp}` are byte-for-byte Git blobs
from c847e6df. Do not format or edit them. `.gitattributes` prevents Windows line
ending conversion. The comparison runner checks all three SHA256 fingerprints;
the baseline executable prints its configured source fingerprint.

## Provenance

| Frozen at | Date | Reason |
|---|---|---|
| a40d37f | 2026-10-09 | First frozen sampler for the behaviour fast path. |
| c847e6df | 2026-10-11 | Release build B6.2: loop breaker periods (think 128, answer 16), final second detection, think EOG guard escalation (43bcbcf3), and the exact max-logit bound for EOG threshold tests (c847e6df). |

The c847e6df freeze followed three checks:

1. c847e6df and 43bcbcf3 give byte-identical transcripts on all 200 cases. The bound
   changes no decision.
2. Against the previous oracle (B6.1 = a40d37f behaviour), the decision-level
   differences are the think EOG guard escalation only. The guard escalates after
   16 blocked end-of-thinking attempts. With escalation disabled, 43bcbcf3 equals
   B6.1 on all 200 cases. The other byte differences are by design: the loop
   ring size, the loop thresholds and the new `guard_escalation` summary line.
3. The B6.2 GPU check passed (probe identity, think-off speed, live loop break,
   two sessions, agent replay identity).

The fixture sets `budget_forced = think_end`. This is the production shape: the
server builds `budget_forced` as the encoded budget message followed by
`think_end`, so it is never empty.

## Use

The baseline and candidate compile the same `tests/core/behaviour_fast_test.cpp`
into separate executables in one CMake configuration. The baseline's include
directory precedes the project root so both its sampler TU and the common
harness (including inline MTP verification) use the frozen types. Its sampling
definitions resolve all sampler references before the static gufo_core archive
is searched; the new sampler's archive member is not pulled into this executable.
There are no old/new sampler definitions in a single translation unit or exe.
JSON grammar sources are unchanged and shared with the candidate.

Run the serial `behaviour_fast_equivalence` CTest or the comparison script with
the two exe paths and an output directory. It compares every transcript byte,
including final distribution double bits, stage before/after distributions,
tokens, RNG, all behaviour-state fields, full counters and request summaries.
Do not compare two candidate executables or use a missing baseline as a pass.
