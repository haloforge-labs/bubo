# Prepare runtime and model files on Windows

`bubo prepare runtime` downloads the three pinned AMD ROCm 10.0.0 wheels for
gfx1151, verifies their SHA-256 values, and installs exactly 13 DLLs plus the
150 rocBLAS and 100 hipBLASLt data files beside `bubo.exe`. The data retain their
`rocblas\library\gfx1151\` and `hipblaslt\library\gfx1151\` subfolders. It
uses no Python, administrator access, GPU, driver installer or model build.
The compiled member inventory is in `tools/release/runtime-members.tsv`.

Bubo does not include AMD files. The runtime step of `bubo prepare` downloads
AMD's ROCm 10.0.0 packages for `gfx1151` directly from `stable.repo.amd.com`.
Before it downloads, it shows the address, size and SHA-256 value of each
package, and the licence information. AMD licenses ROCm per component, and
components can have third-party terms. Read the
[AMD ROCm 10.0.0 licence page](https://rocm.docs.amd.com/en/docs-10.0.0/about/license.html)
and the licence texts that Bubo saves in the `licenses` folder. Other AMD
terms can apply to these files. If you continue, you accept the terms that apply.

The complete text-only release notice bundle, including `NOTICE-windows.md`,
COMGR and hipcc, is saved under `licenses` beside the runtime. Its internal
folders and links are preserved. Its `OPEN-ITEMS.md` records remaining source
and notice provenance questions. This does not assert that no other terms apply.

The archive cache defaults to `ROOT\cache\rocm-10.0.0-gfx1151`. Use
`--cache-dir PATH` for a separate local folder if the default overlaps the
installation. Keep these three wheels for offline repair. `--offline` installs
from a complete hash-valid cache without making a request. A completed runtime
is fully checked, then a rerun does nothing and makes no writes or requests.
Different existing runtime files are preserved and reported with deletion/retry
instructions. A running process that holds the DLLs is named; stop it and retry.
Sharing violations are retried for about ten seconds without disabling AV.

Installation stages and verifies every file, keeps backups during promotion,
and publishes `.bubo\runtime-ready.json` last among installed files. A failed
promotion rolls back the previous runtime. A retained transaction journal is
recovered on the next invocation; interrupted downloads remain resumable.
`--runtime-dir PATH` prepares another installation containing identical
`bubo.exe` and `bubo-engine.dll` files. It does not change the DLL search path.

`bubo prepare model --output D:\BuboData` prepares the fixed eight-file model
package in `D:\BuboData\models\huihui-qwen38-flash-next-ud-q4-k-xl-v1\`.
To use an existing model, place the four shards in
`ROOT\models\huihui-qwen38-flash-next-ud-q4-k-xl-v1\` with the exact catalogue
names. The MTP, vision projector, dense twin and draft head have their own
`mtp\`, `mmproj\`, `twin\` and `draft_head\` subfolders.

Existing main shards are checked by exact size, their GGUF header and tensor
table, and SHA-256 of the first 1 MiB of every tensor (or the whole tensor if
smaller). A saved size, last-write-time and file-identity stamp skips unchanged
main shards on a later prepare run. `--verify` ignores those stamps and checks
the full SHA-256 of all eight files, with progress for each file. Sidecars are
fully hashed on every reuse. New downloads are fully hashed while streaming;
they do not receive an extra full read. A corrupt canonical file is preserved:
the error names it and asks you to delete it and run `bubo prepare` again.

The plan shows each file, its immutable source URL, size, SHA-256, licence,
destination, total and free-space requirement before one download question.
`--yes` prints the same plan and records acceptance. Derived weights are hosted
by the Bubo project and are subject to the model licence; they are not MIT
licensed. A completed rerun prints `prepare: already prepared; nothing to do`
without network requests, acquiring a lock, or filesystem writes. `--offline`
fails on missing files before it makes a request. Interrupted downloads retain a
`.partial` file and a `.partial.json` checkpoint beside the final destination.

Use a local NTFS, exFAT or ReFS volume. FAT32, UNC/network volumes and cloud
placeholders are rejected. Free-space checks reserve the outstanding payload
allocation plus a 2 GiB margin per volume and run again during transfer.

Preparation uses the existing libcurl and Windows certificate store with TLS
validation enabled. It honours `HTTPS_PROXY` and `NO_PROXY`; system PAC settings
are not read. On a connection failure, configure `HTTPS_PROXY=http://host:port`
and run again. Redirects must remain HTTPS and under the reviewed
`huggingface.co` or `hf.co` suffixes. No token or cookie is forwarded.

Production acquisition remains gated on the unresolved Unsloth/Bubo repository
pins and approved pinned model notice bundle. The fixture catalogue and
loopback HTTP exception exist only in the test library. The combined `all`
plan stays gated while model source pins/notices are OPEN; use the independent
runtime command. The combined plan and resolver/startup binding remain J4 work.
