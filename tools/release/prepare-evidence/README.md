# Prepare source records

These records pin source revisions, file sizes and hashes for prepared model packages.
The release build consumes them without network discovery.

To regenerate verified package fingerprints, use the `prepare_fingerprints` target:

```text
prepare_fingerprints INPUT_DIR src/core/prepare/fingerprints --model-set lowmemory --local-evidence tools/release/prepare-evidence
```

Keep the generated fingerprint tables and their input records together. Model attribution is in `tools/release/model-notices/`.
