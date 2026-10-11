# I1 CPU fixtures

All C++ tests use local metadata/byte fixtures only: no socket, network, model,
AMD DLL or GPU. `fixture_catalogue.hpp` and the three HF tree fixtures describe
ASCII `abc` (published SHA-256 known-answer vector) with explicitly synthetic
repository IDs/commits. Production PIN SLOTs remain empty; a fixture cannot
produce a release package. `prepare_gate_test --validate-fixture DIR` exposes
the same validator for a copied evidence set and intentional corruption proof.

`fingerprints.json` is the tiny table fixture. Its envelope binds the logical
shard and complete file digest. Each tensor row has `tensor_name`, absolute
file `offset`, `tensor_bytes`, `checked_length = min(1048576, tensor_bytes)`, and
lowercase `sha256` of the checked prefix. All three lengths/offsets use canonical
decimal strings for exact 64-bit representation. A table is <=1 MiB with
1..4096 unique, nonoverlapping, file-bounded tensors. Unknown/duplicate JSON
fields and noncanonical integers fail. The loader cannot establish GGUF tensor
coverage or compute payload hashes; J2 generates complete production tables
from pinned shards and applies its check policy. Slots are per shard in the
compiled catalogue, with no production prefix hashes invented here.

`prepare-help.txt` is authored from the I1 command contract, never captured
from the executable. The CTest snapshot normalizes only CRLF/LF differences.
The inert engine DLL exercises lazy dispatch, UTF-8 argv, version/revision
identity rejection and exit forwarding. It is a test target, never installed.
`prepare_dependencies_test` checks curl's HTTPS/Schannel metadata without
creating a session, the published OpenSSL SHA-256 `abc` vector and a manually
specified raw-DEFLATE stored block with zlib `inflateInit2(-15)`.
