# Provenance of the lower-memory model files

This composition replaces only the four main GGUF shards. The pinned
composition manifest records 1,080 tensors from the Huihui source and 144
expert tensors from the ISTA source. The composition manifest itself lists
no MTP, mmproj, twin, or draft-head files.

The catalogue retains the Huihui set's MTP, mmproj, twin and draft-head pins
unchanged. In particular, the twin is the Huihui twin, not the original set's
twin-base. This is the declared package recipe; the composition manifest does
not independently attest these auxiliary files.

The four main sizes and whole-file SHA-256 values were measured from the
composed files. Their canonical pins and the manifest digest are in
`src/core/prepare/lowmemory_data.hpp`. The auxiliary files have the same
immutable Hugging Face provenance as the Huihui set.

The four shards are published in the repository
`HaloForgeLabs/bubo-qwen3.8-flash-next-lower-memory`, revision
`2f4f068840c7d85c7b0d8ac044db9e410687d636`, which includes README.md, NOTICE and
LICENSE. The shards and composition manifest are unchanged. Bubo downloads the
shards from that revision and checks each file against the pinned size and
SHA-256 value. It also accepts matching local files.

The published names are `Qwen3.8-Flash-Next-Huihui-LowMemory-0000N-of-00004.gguf`.
The source files were named
`Qwen3.8-Flash-Next-Hybrid-GSQ-RCO-Q4KXL-0000N-of-00004.gguf` (N = 1..4).

LICENSE and NOTICE are byte-for-byte copies of the files in the pinned
model repository revision named above.
LICENSE is the Qwen Community License 1.0 text. NOTICE names Qwen, Huihui,
Unsloth, ISTA-DASLab and HaloForge Labs, and describes the composition.

SHA-256 of the copied bytes:

- LICENSE: `a0dc422560841fd68e06d974907f8b4c709bca44a67daad2b528437bdf676c08`
- NOTICE: `76ca2a43af1ec76d88a0369d5385309b8bc03cf276c11df9fd42915a5acc897b`
