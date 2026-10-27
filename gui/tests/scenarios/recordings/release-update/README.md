# V12-8 evidence inputs

`bubo-release.json` is the offline release builder's actual output for
the `bubo-v0.1.1-windows-x64-gfx1151.zip` release candidate archive,
with required explicit runtime-changed=false, runtime-download-bytes=0,
models-changed=false and models-needs-prepare empty. The all-reuse CLI recordings
below independently support those payload decisions. Its real 0.1.1
package is 66,131,579 bytes; runtime is `rocm-10.0.0-gfx1151`, unchanged with
zero download bytes; models are unchanged with an empty prepare list.

`published-build-manifest.json` is copied verbatim from the extracted published
`BUILD-MANIFEST.json` in the published `bubo-v0.1.0-windows-x64-gfx1151.zip` archive.
It identifies 0.1.0, revision `3ff74dce`, engine 0.10.0. The archive itself is
44,096,931 bytes, SHA-256
`d1c92dacf68e8e379b8e02c8d68ba18ba102b2fd22784e25cb06b0757fac1306`.
The old executable is never run.

The builder's required summary input is the first release WHATSNEW bullet,
without Markdown emphasis. The scenario generator displays this actual manifest
summary unchanged. A transfer in progress is an
arranged state at one quarter of this actual package's byte count.

The job's external capture script produces the `legacy-*` recordings, offline,
from a scratch reconstruction of the published prepare layout and existing
pinned payload hard links. It retains commands, executable hashes, original
package identity and before/after model payload identities and timestamps.
`go run ./update-reuse-view -output <external-path>` transports those records through the
actual public supervisor `ViewBuilder` and checks both zero-byte facts.

External `workflow-*.json` screenshot inputs are retained Views of
`TestTightRestartSavedCheckboxWorkflow`, which exercises the real supervisor
and its test CLI. Its memory estimates, versions, names, contexts and sizes
come from the merged CLI's V12 recordings. The lifecycle is arranged; this is
not a GPU inference or actual GUI-window observation.

Only true recordings belong here. Generated Views and workflow snapshots stay
in the evidence folder outside the repository. Screenshot runs receive those
snapshots through `BUBO_SHOTS_INPUT_DIRECTORY`; no job path is built into tests.
