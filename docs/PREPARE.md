# Prepare runtime and model files on Windows

After a Bubo update, `bubo prepare` reuses runtime members, model shards and side files that still match their pinned versions. It refreshes Bubo's license notices and readiness markers without downloading or extracting those payloads again. If preparation is interrupted, run it again to recover. A damaged payload still follows the normal verification and repair rules, and links to files elsewhere are never replaced.

Text clients see the model's package ID, such as `huihui-qwen38-flash-next-ud-q4-k-xl-v1`, or an imported GGUF's file name without `.gguf`. `--served-model-name` overrides that name. `/v1/models` lists the served name first and `bubo` as a compatibility alias; either name works in requests. Use the real ID in client configurations so the model is identifiable. The Flash Next default profile uses 200,000 context tokens per session; `--context 262144` still selects its native maximum, and other model families keep their own defaults.

Install the VC++ v14 x64 Redistributable
(https://aka.ms/vs/17/release/vc_redist.x64.exe) before these PREPARE steps.
It is a prerequisite and is never shipped or downloaded by Bubo. There is no
checking script entry point for prepare: `bubo.exe` links the CRT dynamically,
so direct `bubo.exe prepare` with the CRT missing shows a Windows loader error
before `main()`. The in-process check only adds protection when some CRT parts
are missing. `start.cmd` and `tools\windows\start.ps1` check before launching
the executable and can print the installation link.

`bubo prepare runtime` downloads the three pinned AMD ROCm 10.0.0 wheels for
gfx1151, verifies their SHA-256 values, and installs exactly 13 DLLs plus the
150 rocBLAS and 100 hipBLASLt data files into the shared runtime folder
`%LOCALAPPDATA%\Bubo\runtime\rocm-10.0.0-gfx1151` (one folder per runtime
version, per user, shared by every Bubo package folder). The data retain their
`rocblas\library\gfx1151\` and `hipblaslt\library\gfx1151\` subfolders. It
uses no Python, administrator access, GPU, driver installer or model build.
The compiled member inventory is in `tools/release/runtime-members.tsv`.

Bubo does not include AMD files. The runtime step of `bubo prepare` downloads
AMD's ROCm 10.0.0 packages for `gfx1151` directly from `stable.repo.amd.com`.
Before it downloads, it shows the address, size and SHA-256 value of each
package, and the license information. AMD licenses ROCm per component, and
components can have third-party terms. Read the
[AMD ROCm 10.0.0 license page](https://rocm.docs.amd.com/en/docs-10.0.0/about/license.html)
and the license texts that Bubo saves in the `licenses` folder. Other AMD
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
`--runtime-dir PATH` installs the runtime in another absolute local folder. The
folder needs no copy of `bubo.exe` or `bubo-engine.dll`. Bubo does not remember
that folder: set `BUBO_RUNTIME_DIR` to it before you run `bubo`. The last line of
the output gives the exact value (`BUBO_RUNTIME_DIR=<folder>`), and the machine
`result` carries it as `runtime_environment`. `--output ROOT`
moves the models and the archive cache, not the runtime.

Where `bubo.exe` looks for the runtime, in this order:

1. `BUBO_RUNTIME_DIR`, if set (an absolute local folder). Bubo uses it and
   reports an error if it holds no valid runtime; it does not fall back.
2. `%LOCALAPPDATA%\Bubo\runtime\rocm-10.0.0-gfx1151`, if its
   `.bubo\runtime-ready.json` marker is valid.
3. The folder that holds `bubo.exe` (the v0.1.0 layout), if it holds a valid
   runtime. Such a runtime keeps working and is not moved or deleted.

Before it loads `bubo-engine.dll`, `bubo.exe` adds the chosen folder to the
DLL search path, so the AMD DLLs resolve from it. rocBLAS and hipBLASLt find
their `library` folders next to their own DLLs, in the same folder. A user
never sets a search path or a tensile path. `bubo status --json` reports
`paths.runtime`, `paths.runtime_kind` (`shared`, `explicit` or `legacy`) and
`paths.runtime_shared`.

Migration: a v0.1.0 install keeps its runtime beside `bubo.exe` until
`bubo prepare runtime` installs the shared folder (from the archive cache with
no download, if the cache is complete). Then the server uses the shared folder.
Bubo never deletes the old files; remove them by hand when you want the space.

`bubo prepare model` prepares the default eight-file model package in
`%LOCALAPPDATA%\Bubo\models\huihui-qwen38-flash-next-ud-q4-k-xl-v1\`. With
`--output E:\Bubo` it uses `E:\Bubo\models\...` instead, and `bubo serve`
finds it there.
To use an existing model, place the four shards in
`ROOT\models\huihui-qwen38-flash-next-ud-q4-k-xl-v1\` with the exact catalog
names. The MTP, vision projector, dense twin and draft head have their own
`mtp\`, `mmproj\`, `twin\` and `draft_head\` subfolders.

`bubo prepare model --model-set original` prepares the original model, which is
not abliterated, in `ROOT\models\qwen38-flash-next-base-ud-q4-k-xl-v1\`. It has
the same eight files, the same checks and the same resume rules. The four shards,
the MTP file and the vision projector come from `unsloth/Qwen3.8-Flash-Next-GGUF`
at revision `766911a6b7369840a91dbcd95f9f997acaab6cd6`. The dense twin and the
draft head come from `HaloForgeLabs/bubo-qwen3.8-flash-next` at revision
`0a3c343bec2095f43f2906371c83e4faa8a47da1`. The plan lists every file, size and
SHA-256 value before any download. The original set has its own `LICENSE`,
`NOTICE` and `PROVENANCE.md` in its `licenses` folder. The saved default model
stays the Huihui package. Start the original set with
`--manifest ROOT\models\qwen38-flash-next-base-ud-q4-k-xl-v1\bubo-package.json`.

Existing main shards are checked by exact size, their GGUF header and tensor
table, and SHA-256 of the first 1 MiB of every tensor (or the whole tensor if
smaller). A saved size, last-write-time and file-identity stamp skips unchanged
main shards on a later prepare run. `--verify` ignores those stamps and checks
the full SHA-256 of all eight files, with progress for each file. Sidecars are
fully hashed on every reuse. New downloads are fully hashed while streaming;
they do not receive an extra full read. A canonical file that no longer matches
its pin is planned as a download ("replaces a file that did not match"). After
you accept the plan, Bubo renames it aside (`<name>.damaged-<UTC time>`), downloads
and verifies the file again, and then removes the renamed copy. Bubo never writes
into the damaged file, so other hard links to it keep their bytes. A file you
supplied with an option, or one whose real location is outside the package
folder, is still refused (exit 6).

The plan shows each file, its immutable source URL, size, SHA-256, license,
destination, total and free-space requirement before one download question.
`--yes` prints the same plan and records acceptance. Derived weights are hosted
by the Bubo project and are subject to the model license; they are not MIT
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
