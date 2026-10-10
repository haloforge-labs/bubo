# Release switch generator

Run `python tools/release/gen_switches.py` after changing the JSON catalog.
Run the same command with `--check` to detect committed output drift.
Python is not used to generate files during a build.

Each inventory group expands into individual named rows.
Live no-op controls remain options; their profile values stay null.
Removed names appear in catalog metadata but never in generated environment reads.
Other model controls remain available under their own model family.

The schema adds four fields to the requested descriptive row:

- `built_in` records the reader default for the settings echo. It is never injected.
- `value_rule` selects a closed value validator.
- `aliases` records parser short options for the startup snapshot.
- `takes_value` records parser arity so option values cannot become modality names.

A null subagents value leaves the reader default in effect.
A null exact value inherits the default profile value.
A null default leaves the reader default in effect.
Stock applies no profile or manifest defaults.
The closed `requires` vocabulary includes artifact, session, speculative, and thinking conditions.
`think!=off` skips the medium effort default when an explicit flag disables thinking.
A model-free settings echo selects Qwen3.8 Flash Next for inspection.
A model path causes header-only architecture and native-context inspection.

The C++ test regenerates both committed outputs without Python.
The docs renderer also compares with `tests/fixtures/release_flags_golden.md`.
The Python check compares that golden, proving agreement with the C++ check.
Update the golden with reviewed catalog changes, then run both checks.

`ManifestArtefacts` remains empty for P1.
`ReleaseInputs::manifest_defaults` permits simulated precedence checks without a manifest file loader.
Production leaves this map empty.

Set the CMake variable `GUFO_PRODUCTION_ENV_SPEC` to a production JSON path.
Its initial default can come from the environment with the same name.
An empty path produces a clear skip; the committed fixture always runs.
The CPU test also accepts `--production-spec PATH` for an additional comparison.
