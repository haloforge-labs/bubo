# Remaining platform and verification gaps

- `tests/cache/backing_test.cpp`: retained on Linux; its seccomp filters,
  madvise failure injection and ELF `--wrap` symbols cannot build as a native
  Windows executable. A native Windows fault injector is needed for this GPU
  backing test. Other native component ledger/checkpoint tests remain enabled.
- `hip_memory_visibility_restricted_test`: retained and compiled; Windows has
  no POSIX process memlock limit, so its restriction-only variant reports skip
  77 there. Ordinary memory visibility variants remain enabled. No GPU variant
  was executed in this job.
- `qwen27b_continuation_adapter_test`: retained on Linux; its RTLD_NEXT
  submission/completion fault interception needs native Windows fault seams.
- `qwen_dflash_upload_lifetime_test` and `qwen_vision_prepare_test`: retained on
  Linux; ELF `--wrap=hipMemcpyAsync` needs a native Windows upload observation
  and failure-injection seam.

CPU test exclusions and native-Windows tests absent from CTest are recorded
individually in the job report; none counts as passing.

## Component-cache acceptance port

Fast mode is enabled through its existing expert/tail knobs. B is an ordinary
checkpoint; the component ledger and retention policy own capacity and eviction.
The original tests remain in the tree. The component fixtures replace these
obsolete assertions from `--tail-newest 0..13`:

- 0: source retirement becomes ordinary shared-chunk publication; B has no retired-source class.
- 1: retired-source destruction becomes a real capture-transfer failure; ordinary owners survive failed copies.
- 2: hand-built snapshot size mismatch becomes policy-identity separation; component allocation supplies actual sizes.
- 3: fixed branch priority becomes divergent-token reuse; upstream retention owns ranks.
- 4: concurrent retired-source state becomes two-slot preservation; slots hold immutable borrowed rows.
- 5: foreign-LRU victim selection becomes idle-peer history; a specific eviction victim is no fast-mode contract.
- 6: completed-foreign special status becomes an ordinary completed peer; no newest-B record class exists.
- 7: special leased status becomes a held ordinary peer; real slot ownership supplies the guarantee.
- 8: zero-host-memory synthetic admission becomes actual 4 MiB ledger pressure; backing and metadata are charged.
- 9: own-lineage special priority becomes an ordinary frontier; B uses normal continuation retention.
- 10: tail-final retirement becomes rewritten final rows; numerical provenance must reject stale reuse.
- 11: special tail-final records become same-history exact hits; normal checkpoint identity supports retry reuse.
- 12: hand-made depth/publication races become real allocation failure; the one-third/depth caps were removed.
- 13: fixed foreign-LRU order becomes distinct foreign histories; upstream retention owns removal order.

The historical stdout/eviction SHA256 values are retained in
`tests/fixtures/prefill_tail_default_goldens.sha256` and the legacy fixture.
They are not claimed as verified by the new component fixture: its default
golden compares executed inactive/tail-zero requests with an independent
identity-disabled control, including selected tokens, rows and ledger bytes.

`video_jobs_test` and `serving_bench_test` require symlink fixtures. This host
refuses symlink creation (Windows privilege error 1314); they report skip 77
instead of passing. Run them on a host with symlink creation enabled.
