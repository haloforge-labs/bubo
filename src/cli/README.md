# CLI (`src/cli`)

This directory contains the unified `bubo` CLI.

## Philosophy

1. **One Binary**: All user-facing capabilities (`serve`, `prompt`, `chat`, `bench`, `video`, `diagnose`, `probe`) live under the single `bubo` executable.
2. **Subcommands Over Flag Soup**: Model-specific or modality-specific options belong under explicit subcommands (e.g. `bubo serve ... llm ...`, `bubo serve ... video ...`), keeping global options clean and self-documenting.
3. **Declarative Parsing**: Flags and arguments are declared and bound to typed struct fields using `ArgParser` (`src/cli/arg_parser.hpp`). Never parse raw `argv` manually.
4. **CLI Layer vs Engine Core**: Code in `src/cli/` handles argument parsing, terminal output, and daemon lifecycle. Heavy computational work and kernel dispatches belong in `src/models/` and `src/core/`.

## Directory Organization

Each top-level command has its own dedicated subdirectory:

```
src/cli/
├── main.cpp              # Root dispatcher & global flags (-h, -v, help)
├── arg_parser.hpp        # Declarative type-safe argument parser
├── bench/                # `bubo bench` (model throughput & kernel benchmarks)
├── diagnose/             # `bubo diagnose` & `bubo probe` (hardware inventory & toolchain probe)
├── prompt/               # `bubo prompt` & `bubo chat` (text generation CLI & REPL)
├── serve/                # `bubo serve` (HTTP API server: `llm`, `video`, `audio`)
└── video/                # `bubo video` (MiniMax H3 text-to-video generator CLI)
```

## Adding a Command or Flag

- **Adding a Flag**:
  - Add a field to the command's options struct.
  - Bind it in the command's `ArgParser` definition using `parser.AddOption(...)` or `parser.AddFlag(...)`.
  - **Group Organization**: Place the flag in a logical group (e.g. `Model`, `Sampling`, `Hardware`, `Speculative`, `Diagnostics`). If no existing group fits, create a new meaningful group.
  - **Descriptions**: Keep descriptions concise yet explanatory (include defaults and expected value formats).
  - **Partial Implementation Notation**: If a flag is not yet supported for all models/backends, annotate the description with `(TODO: <model>)` (e.g., `(TODO: deepseek)` or `(TODO: qwen)`).
- **Adding a Command**:
  1. Create `src/cli/<command>/<command>.hpp` and `<command>.cpp` in `namespace gufo::cli`.
  2. Implement `Run<Command>(std::span<const char* const> args)`.
  3. Register its name, summary, engine requirement and runner in `src/cli/command_table.inc`; bootstrap, engine dispatch and global help share that table. Add the uniform-context runner adapter beside the engine adapters in `main.cpp`, or the engine-free adapters in `command_table.cpp`.
  4. Before the code, write the ways the command can fail and an end-to-end test that runs the real `bubo.exe` (`tests/cli/`). Then add the source files to `CMakeLists.txt` under `add_executable(gufo ...)`.
