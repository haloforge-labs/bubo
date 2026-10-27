# Published 0.1.0 metadata fixture

`published-v0.1.0.json` comes from the local published archive
the published `bubo-v0.1.0-windows-x64-gfx1151.zip` release archive.
The archive SHA-256 is
`d1c92dacf68e8e379b8e02c8d68ba18ba102b2fd22784e25cb06b0757fac1306`;
its BUILD-MANIFEST identifies clean revision
`3ff74dce9a67182e572dbddbd8279c058e60fe31`, product version 0.1.0.
The executable was read as bytes and was never executed.

The 263-row runtime table is extracted directly from the executable's embedded
`bubo-runtime-members-v1` string. All 70 runtime notice texts and hashes were
decoded from that revision's `runtime_notices.inc`, checked against SHA-256,
and individually confirmed present verbatim in the archive's executable.

The published zip contains no AMD runtime payload, runtime readiness marker,
model package marker, or model prepare-state file. `runtime_ready` reconstructs
the output of that revision's `Ready` constructor from its embedded table and
notice pins. `model_profiles`, `prepare_state`, and `prepare_state_row_fields`
record its `Manifest` and `State` constructors in `model_prepare.cpp`.
File stamps depend on the installation and are populated by the CPU fixture.

The runtime test compares every real member path, size and hash with today's
compiled pins, checks every old notice hash, and passes the old marker through
the production readiness reader before the missing-payload check. The model
test applies the published marker/state shapes to eight tiny synthetic payloads,
then checks offline metadata refresh, unchanged transfer count, exact current
marker bytes, and preserved payload hashes and timestamps. This is a CPU
metadata compatibility test, not a claim of hashing a prepared 1 GiB AMD runtime
or a 100 GiB model installation.
