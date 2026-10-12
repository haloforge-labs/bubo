# ROCm 10.0.0 gfx1151 member inventory

`runtime-members.tsv` and `src/core/prepare/runtime_members.inc` are generated
from the three official AMD wheels. Archive sizes and SHA-256 values are from
`<local path>`,
Appendix B. The 13 DLL hashes are checked against Appendix A and
`<local path>`.
URLs follow the binding `PREPARE-CONSENT-TEXT-20261011.md` (including the
`/rocm/core/whl-next/` segment), superseding the older design URLs.

The CPU-only native tool takes three wheel files on disk and two output paths:

```text
prepare_runtime_table CORE.whl LIBRARIES.whl DEVICE.whl TABLE.tsv EMBEDDED.inc
```

Production generation checks the archive pins before reading selected members,
then streams exactly 13 DLLs, 150 rocBLAS data files and 100 hipBLASLt data files
through linked zlib and OpenSSL. It checks ZIP CRC and each Appendix A DLL hash.
It never downloads, executes wheel contents, uses Python or loads a GPU.
The `--fixture` tool flag accepts tiny synthetic archives for developer tests;
it is not a preparation command or a production pin override.

The table is complete: 263 rows, 317,324,825 uncompressed bytes. Original
read-only inputs were the named files under the earlier report's
`dl/rocm-sdk-{core,libraries,device-gfx1151}-10.0.0/` directories. Their full
archive hashes matched Appendix B before generation. Each row contains the
wheel index, exact member path, installation-relative destination, exact
uncompressed length and SHA-256. Installation preserves the `gfx1151`
subfolders. The engine validates the compiled table schema, counts, paths,
duplicates, limits and DLL pins. It does not accept a user-supplied table.

This closes the missing-inventory portion of O1. Independent review, real AMD
HTTPS/Range/proxy qualification and the notice bundle's recorded source/build
provenance gaps remain release work; generation does not claim to close them.
