# Prepare the model files on Windows

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
loopback HTTP exception exist only in the test library. The runtime and `all`
steps remain pending J3; resolver/startup binding remains J4 work.
