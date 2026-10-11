# Prepare source evidence (OPEN)

The owner-authorized provenance step supplies `huihui.json`, `unsloth.json`,
and `bubo.json` here. No network discovery occurs during builds or preparation.
The production Unsloth/Bubo commit slots and Bubo repository sentinel are in
`src/core/prepare/release_catalogue.hpp`. They remain OPEN in I1.

Each UTF-8 file (maximum 1 MiB) has this envelope:

```json
{
  "schema_version": 1,
  "repository": "<source repository ID>",
  "commit": "<full immutable 40 lowercase hex commit>",
  "tree": [
    {"type": "file", "path": "<repo-relative payload path>",
     "size": 123, "lfs": {"oid": "<64 lowercase hex SHA-256>", "size": 123}}
  ]
}
```

`tree` preserves saved HF tree entries at that commit, including `path`, `size`
and `lfs.oid`. Additional HF fields are allowed inside tree entries and LFS
objects. Directory entries and ordinary Git notice/card files are allowed.
Every pinned payload must be a file with matching path, size and LFS SHA-256.
No repeated path/repository or commit mismatch is accepted. JSON duplicate
keys and unknown envelope fields fail. Integer sizes must be exact (<=2^53-1).
Do not substitute metadata from a branch, tag, different revision or download.

`prepare_gate --release DIR` checks the compiled production catalogue against
all three files offline. `prepare_release_check` is the explicit CI target.
CMake's `Release` configuration invokes the same gate when building bubo or
the default target. Development configurations allow OPEN pins and have a
separate `prepare_gate --development` field check. Fixture tests use their own
tiny catalogue; no fixture override is compiled into the release gate.

This I1 gate covers model pins and source-identity evidence only. Approved
licence/notices, the 263-member runtime table, complete shard fingerprints,
actual redirect/resume/TLS qualification and final HIP release proof remain
separate J2+ release requirements. The existing package scanner stays enabled.
