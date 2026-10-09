# Frozen sampler oracle

`src/core/{sampling.cpp,sampling.hpp,behaviour.hpp}` are byte-for-byte Git blobs
from a40d37f. Do not format or edit them. `.gitattributes` prevents Windows line
ending conversion. The comparison runner checks all three SHA256 fingerprints;
the baseline executable prints its configured source fingerprint.

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
