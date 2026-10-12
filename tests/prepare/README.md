# Prepare CPU fixtures

The I1 tests use local metadata/byte fixtures with no sockets, AMD DLL or GPU.
`fixture_catalogue.hpp` and the three HF tree fixtures describe
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
fields and noncanonical integers fail. The schema-1 loader establishes format
only. Schema 2 pins the generator version, complete GGUF header SHA-256 and
length, names, types, shapes, relative/absolute offsets, tensor byte lengths and
every tensor-prefix hash. The four production tables in
`src/core/prepare/fingerprints/` are compiled into the library; mutable external
JSON cannot replace them. Metadata-only shards require the header pin to cover
the whole file. Source hashes were checked before generating these tables.

`prepare-help.txt` is authored from the command contract, never captured
from the executable. The CTest snapshot normalizes only CRLF/LF differences.
The inert engine DLL exercises lazy dispatch, UTF-8 argv, version/revision
identity rejection and exit forwarding. It is a test target, never installed.
`prepare_dependencies_test` checks curl's HTTPS/Schannel metadata without
creating a session, the published OpenSSL SHA-256 `abc` vector and a manually
specified raw-DEFLATE stored block with zlib `inflateInit2(-15)`.

J2 adds separate production-policy and test-injection libraries. Production
`HostPolicy` has no loopback member; its policy executable proves HTTP loopback
is rejected. Only `bubo_prepare_test` defines `BUBO_PREPARE_TESTING`. The model
catalogue override, loopback exception and native stub are absent from the
shipped bootstrap.

The native stub binds 127.0.0.1 on an OS-assigned port excluding 8081-8083. Its
HF-layout mirror and generated streams exercise actual curl callbacks, Range
and response handling without contacting external hosts. The model test also
acquires eight valid tiny GGUF files through the stub. Fake transports exercise
HTTPS suffixes, credential header rebuilding, pre-checks and retry policy.

Model and loopback CTest wrappers choose fresh folders under the build
directory. For direct execution, supply `prepare_model_test` an unused fixture
root and `fixtures/model-consent.txt`; set `BUBO_PREPARE_TEST_ROOT` for
`prepare_loopback_test`. Native tests retain tiny evidence fixtures. Every
build/run requires the operator's owned build lock and BelowNormal priority.

`model_mutations.cmake` and `transport_mutations.cmake` copy actual source with
mandatory anchors, remove one guard, and link the same intact tests. Mutants
must fail. They test detection of broken suffix policy, streaming hash,
X-Linked pre-check, FAT32 rejection, manifest-last, no-op writes and consent.
