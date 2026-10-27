# CLI contract, schema version 1

Use the installed `bubo.exe` by its full path. This document specifies the
machine commands from J1a through J2b, at `schema_version` 1. Examples use
fixture data. A contract freeze does not approve a release package or inference.
Run the CPU contract proof and the separate manual release checks.

## Wire rules

Send and receive UTF-8 without a BOM. End each record with LF. Each record is
one JSON object. Escape CR, LF, quotes, and backslashes inside strings. Accept
unknown output fields. Reject an unsupported `schema_version`.

For every machine command, stdout contains only protocol records: no prose,
ANSI colors or terminal progress redraws. Human diagnostics and debug logs go
to stderr or a named log file. Read stdout and stderr concurrently.

Consumers ignore unknown event types for state changes and log the event name,
operation identity and sequence for diagnosis. Still validate their common
envelope, version, framing and ordering. Unknown fields remain additive on
known events; a malformed known payload is a protocol error.

Output records, including LF, are at most 1 MiB. Prepare splits a large plan
into numbered component records. Input records contain at most 65536 bytes
before LF. A partial record waits for LF. A partial record followed by EOF is
invalid input. Garbage, unknown input fields, wrong identity, stale approval,
and a second approval fail with `invalid_arguments`, exit 2.

A failed stdout write disables that writer. A blocked native write has a
2 second deadline. Prepare cancels when its output fails. A truncated final
record cannot be parsed. Retain the raw process exit and last complete event.
The CLI does not inherit its saved machine stdout handle into a child.
Prepare also checks stdout while it waits for approval. Closing stdout alone
cancels the operation when stdin remains open.

Every output record has these fields:

| Field | Rule | Example value |
|---|---|---|
| `schema_version` | Integer. Current version is 1. | `1` |
| `type` | Event type from the list below. | `"plan"` |
| `operation_id` | 32 lowercase hexadecimal characters. Binds input to this operation. | `"11111111111111111111111111111111"` |
| `seq` | Positive integer. Starts at 1. Increases by 1 for this writer. | `1` |
| `time` | UTC time, with seconds and a Z suffix. | `"2026-10-13T18:00:00Z"` |
| `product` | Product identity object. | `{"name":"bubo","version":"0.1.0","revision":"fixture","engine_version":"0.1.0"}` |
| `payload` | Event data object. | `{}` |
| `name` | Product name. | `"bubo"` |
| `version` | Product version. | `"0.1.0"` |
| `revision` | Source revision. Can be empty or unknown in an archive build. | `"fixture"` |
| `engine_version` | Engine version. | `"0.1.0"` |

JSON counters and sizes use exact integers from 0 to 9007199254740991.
Sequence numbers start at 1. Byte fields use bytes. An unavailable value is
null, not zero. Check the event and command before you interpret an exit code.

When an `error` event's typed `code` disagrees with the command's numeric exit
mapping, use `code`, `retryable` and `remedy` for diagnosis and user guidance.
Retain the raw process exit separately and log the disagreement. An error event
is a failure even if the process exits 0; a nonzero process exit cannot become
success because an earlier event looked successful. Readiness is independent:
exit 0 does not promise a ready runtime or model. The documented optional NPU
side-job exit 11 can accompany `prepared: true` without an error event.

## Commands and flags

Use `--jsonl` for Prepare. Keep stdin open through completion, including with
`--yes`. Status and Diagnose accept `--json` for one result object. Their
`--jsonl` success stream starts with `capabilities`. Early failures can emit
one error record before capabilities. Prepare accepts `--json` with the
same stream as `--jsonl`. Select only one machine flag.

This is the output-shape matrix; the following table is the command/option
reference. A stream may contain just one record on early failure.

| Command | `--json` stdout | `--jsonl` stdout | Without a machine flag |
|---|---|---|---|
| Global help/version; command machine help | Single `capabilities` object | Single `capabilities` object | Human help/version |
| `help TOPIC` | Usage error | Usage error | Human help |
| `prepare [all\|runtime\|model]` (including `--plan`) | JSONL stream (compatibility) | JSONL stream | Human progress/result |
| `status` | Single `status` object | `capabilities`, then `status` | One line with the next step or running server |
| `status --watch` | Repeated `status` objects (compatibility) | `capabilities`, then repeated `status` objects | Live prompt and response counts per session |
| `diagnose` (including artifact modes) | Single report/artifact object | `capabilities`, then report/artifact | Human report |
| `diagnose --bundle` / `--preview` | Single `support_bundle` object | Typed usage error | Typed usage error |
| `models list` / `models estimate` | Single models/estimate object | `capabilities`, then models/estimate | Human list/estimate |
| `serve --list-switches` | Single `switches` object | Typed usage error | Typed usage error |
| `serve` | Usage error | Lifecycle/request JSONL stream (Windows control) | Human server logs |
| `stop` | Acknowledgment JSONL stream (`stopping`, `stopped`) | Usage error | Usage error |
| Other commands | Typed unsupported error | Typed unsupported error | Existing human output |

Use `status --jsonl --watch` for a watch stream. Its first successful record is
`capabilities`; subsequent `status` records share the operation ID and increase
`seq`. Keep stdin open while watching. Closing stdin ends the stream with exit
0 and no terminal `result` event. A failed stdout write or control query ends
the command early; retain its last complete event and raw exit. The existing
`status --json --watch` spelling remains supported: it also emits JSON Lines,
but omits the initial capabilities record.

For a person, use `bubo status --watch` (optionally `--instance-id ID`). In a
terminal it redraws one line per session, labelled "reading" or "writing".
Reading shows a bar, percent, read/total tokens, rate of new work, approximate
ETA, and finally any cached tokens. CLI and GUI use the same validity rule:
total, cached and prefilled must be nonnegative safe integers, total must be
positive, cached <= total, and prefilled <= total - cached. Read = cached +
prefilled; percent = floor(100 * read / total). Invalid or missing counts show
progress unavailable. Counts advance only when the server reports them.

Reading rate uses the first observed count/time and the latest increased
count/time for this session, request and state. Unchanged polls retain that
estimate. Until the count increases, rate and ETA are dashes. A new request,
state change, counter decrease or instance change resets the baseline. Cached
tokens count as read, never as new work. Writing keeps the adjacent-observation
rate and shows generated tokens without percent or ETA, since length is unknown.
Control-call failures show "Server not answering" and polling continues.

Polls run once per second. With redirected stdout, print a plain line only when
it differs from the previous line (sessions separated by ` | `). Ctrl+C or stdin
EOF stops the watch; a second Ctrl+C uses the operating system's interrupt
behavior. `NO_COLOR` is respected; this view uses no terminal colors.

| Command or flag | Behavior | Example command line |
|---|---|---|
| Global `--help`, `-h` | Machine mode returns capabilities. | `bubo --help --json` |
| Global `--version`, `-v` | Machine mode returns capabilities and envelope product identity. | `bubo --version --json` |
| `help TOPIC` | Human help only. A machine flag makes this a usage error. | `bubo help prepare --json` |
| `prepare [all]` | Preview and prepare runtime and model. Default step is all. | `bubo prepare all --jsonl` |
| `prepare runtime` | Prepare runtime only. | `bubo prepare runtime --jsonl` |
| `prepare model` | Prepare the pinned eight-file Huihui package only. | `bubo prepare model --jsonl` |
| `--jsonl` | Emit JSON lines. | `bubo prepare --jsonl` |
| `--json` | Emit a machine result; Prepare still streams. | `bubo status --json` |
| Prepare `--output ROOT` | Model parent folder. Default is `%LOCALAPPDATA%\Bubo`. | `bubo prepare model --output "D:\Bubo data" --jsonl` |
| Prepare `--yes` | Approve the displayed plan for explicit automation. | `bubo prepare model --yes --jsonl` |
| Prepare `--verify` | Force full SHA-256 checks. | `bubo prepare model --verify --jsonl` |
| Prepare `--offline` | Use local files. Fail before network if files are missing. | `bubo prepare model --offline --jsonl` |
| Prepare `--resume` | Resume tool-owned partial files. Default is on. | `bubo prepare model --resume --jsonl` |
| Prepare `--no-resume` | Restart tool-owned partial downloads. | `bubo prepare model --no-resume --jsonl` |
| Prepare `--mtp-model PATH` | Copy an existing file that matches the pinned hash. Rejected for runtime only. | `bubo prepare model --mtp-model "D:\models\mtp.gguf" --jsonl` |
| Prepare `--mmproj PATH` | Copy a projector that matches the pinned hash. Rejected for runtime only. | `bubo prepare model --mmproj "D:\models\mmproj.gguf" --jsonl` |
| Prepare `--model-set SET` | Choose the model files: `huihui` (default), `original` or `lowmemory`. Rejected for runtime only. | `bubo prepare model --model-set original --jsonl` |
| Prepare `--catalogue-id ID` | Prepare one model catalogue entry (see Models and the model catalogue). Flash Next ids select their set; other ids use the generic path. Rejected for runtime only and with `--model-set`. | `bubo prepare model --catalogue-id qwen36-27b-ud-q4-k-xl --jsonl` |
| Prepare `--plan` | Model step: write the plan (files, sizes, free space, licences) and stop. No download, no write. The machine `result` has `plan_only` true. | `bubo prepare model --catalogue-id qwen36-27b-ud-q4-k-xl --plan --jsonl` |
| Prepare `--cache-dir PATH` | Runtime archive cache folder. | `bubo prepare runtime --cache-dir "D:\Bubo cache" --jsonl` |
| Prepare `--runtime-dir PATH` | Runtime destination: an absolute local folder, no copy of the executable needed. Default is `%LOCALAPPDATA%\Bubo\runtime\rocm-10.0.0-gfx1151` (independent of `--output ROOT`). `BUBO_RUNTIME_DIR` sets it when the option is absent, and `bubo serve` and `bubo status` read the same variable. After a successful runtime step with this option, the human output ends with the line `prepare runtime: set BUBO_RUNTIME_DIR=<folder> so other bubo commands find this runtime`, and the machine `result` has `runtime_environment` (`BUBO_RUNTIME_DIR=<folder>`); without the option that field is absent. Rejected for model only. | `bubo prepare runtime --runtime-dir "D:\BuboRuntime" --jsonl` |
| Prepare `--help` | Return capabilities before prerequisite checks. | `bubo prepare --help --jsonl` |
| `models list` | List prepared models, catalogue entries and GGUF files in the model folder and in each `--scan-folder`. Needs the engine library. | `bubo models list --json --scan-folder "D:\models"` |
| `models estimate` | Memory estimate and recommended context for a GGUF path, a folder or a catalogue id. Needs the engine library. | `bubo models estimate --model qwen38-27b-ud-q4-k-xl --json` |
| `status` | Return preparation readiness before engine loading. | `bubo status --json` |
| Status `--watch` | Emit status every 1000 ms; stop when stdin closes. | `bubo status --jsonl --watch` |
| Status `--instance-id ID` | Select the verified instance. | `bubo status --json --instance-id fixture` |
| Status `--manifest PATH` or `--model PATH` | Report readiness and the model facts for that package instead of the saved default. Not both. | `bubo status --json --manifest "D:\models\gemma\bubo-package.json"` |
| `diagnose` | Return the diagnostic report. Can require the engine DLL on Windows. | `bubo diagnose --json` |
| Diagnose `--fingerprint` | Return a fingerprint artifact. | `bubo diagnose --fingerprint --json` |
| Diagnose `--validate-artifact PATH` | Validate a local artifact. | `bubo diagnose --validate-artifact "D:\report.json" --json` |
| Diagnose `--benchmark NAME` | Existing bandwidth or allocation diagnostic. `hip-allocation` is also accepted. | `bubo diagnose --benchmark bandwidth --backends cpu --json` |
| Diagnose `--backends CSV` | Bandwidth backends. Default is cpu,hip. | `bubo diagnose --benchmark bandwidth --backends cpu --json` |
| Diagnose `--warmup N` | Unsigned 32-bit count. Default is 3. | `bubo diagnose --warmup 3 --json` |
| Diagnose `--repetitions N` | Unsigned 32-bit count. Default is 10. Allocation requires at least 1. | `bubo diagnose --repetitions 10 --json` |
| Diagnose `--duration-ms N` | Unsigned 32-bit milliseconds. Default is 2000. | `bubo diagnose --duration-ms 2000 --json` |
| Diagnose `--working-set-mib CSV` | Allocation sizes in MiB. Default is 64. | `bubo diagnose --working-set-mib 64 --json` |
| Diagnose `--output PATH` | Save the diagnostic document. Stdout still has the event. | `bubo diagnose --output "D:\report.json" --json` |
| Diagnose `--section NAME` | all, inventory, platform, or gpu. Default is all. | `bubo diagnose --section inventory --json` |
| Diagnose `--help`, `-h` | Return capabilities in machine mode. | `bubo diagnose --help --json` |
| `serve` | Emit Windows lifecycle events through `--jsonl`. | `bubo serve --jsonl` |
| `stop` | Acknowledge verified graceful stop, then wait for process exit. | `bubo stop --json --instance-id fixture --wait-ms 45000` |

Stop requires `--json` and `--instance-id`. `--wait-ms` defaults to 45000.
It accepts 0 through 4294967294 ms; 4294967295 is rejected because Windows uses it for an infinite wait. Other commands, including prompt, bench, eval, video, transcribe,
chat, and probe, have no part 1 machine contract. A machine request to one of
them returns a typed unsupported error with exit 2. Human commands keep their
existing output and exit codes. Prepare accepts `--model-set huihui|original|lowmemory`.
The default is `huihui`. The `original` set prepares the unsloth shards and the
HaloForge Labs twin and draft head, with its own legal texts, in the folder
`models\qwen38-flash-next-base-ud-q4-k-xl-v1`. The result record names its
`manifest_path`. Prepare does not change the saved default model for `original`.
Start it with `--manifest PATH`.

Prepare paths must be nonempty. They cannot start with a hyphen or contain
CR, LF, tab, or NUL. Path options cannot repeat. Steps must precede flags.
Diagnose accepts both `--flag VALUE` and `--flag=VALUE` for value flags.

## Prepare input and plan assembly

Wait for `approval_required`. Use its `operation_id` and combined
`plan_digest`. The combined digest binds the component digests. Approve only
after you receive all parts and show the full plan to the user. If the plan
changes, the CLI sends a new plan and requires a new approval.

| Input field | Rule | Example value |
|---|---|---|
| `action` | approve, decline, or cancel. No unknown input fields. | `"approve"` |
| `operation_id` | Copy the operation ID from the event. | `"11111111111111111111111111111111"` |
| `plan_digest` | Required for approve and decline. Copy the approval event digest. Optional for cancel. | `"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"` |

Example input lines, one for each action:

```json
{"action":"approve","operation_id":"11111111111111111111111111111111","plan_digest":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"}
{"action":"decline","operation_id":"11111111111111111111111111111111","plan_digest":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"}
{"action":"cancel","operation_id":"11111111111111111111111111111111"}
```

Cancel before approval or during work returns exit 3. EOF during preflight
verification also cancels. EOF while approval is
required declines. EOF after approval cancels, including final verification.
A prepared no-op needs no approval and can succeed with closed stdin. Keep
stdin open when using `--yes`. A second approval is invalid input, exit 2.

Each component has its own `plan_digest`, `part`, and `parts`. `part` starts
at 0 and must be less than `parts`. All part counts and digests must agree.
Collect records in order. The header has empty arrays and an empty paths map.
Append source and volume records. Merge path records. For each licence path,
concatenate text at the given UTF-8 byte offset. Require the complete byte
count and all numbered parts before approval. A record with an indivisible
field larger than the event limit fails with prepare_internal, exit 9.

The digest covers the full component plan before splitting. It excludes the
`plan_digest` itself and observed `free_bytes`. It includes paths, source
identities, states, legal text, and required space. Use the CLI digest. Do not
compute a replacement digest in the client.

## Event fields and examples

The examples below use the common envelope. Serve events carry process identity independently of log verbosity. The
examples are fixtures; native test transcripts are separate evidence.

### Event `error`

A startup `invalid_arguments` error may include `switch`, the registry row name
rejected by release-settings resolution. Code and remedy retain their meaning.

Failure data: `code` is the stable error string; `component` identifies the source; `stage` identifies the phase; `retryable` is a boolean; `remedy` gives a safe action. All five fields are required.

`detail` is optional: the cause in the command's own words, at most 4096
characters, the same text the command writes to stderr. Prepare and serve
errors fill it when the cause is known, for example the file that does not
match its pin, the drive that has too little space with the numbers, or the
model file that is missing. Show it with the remedy. Do not parse it; decide
on `code`. Prepare takes it from its last `prepare: E<exit> ...` stderr line
for that exit value; serve takes it from the exception or its last `Error`
stderr line. In machine mode serve now writes startup errors to stderr too.

```jsonl
{"schema_version":1,"operation_id":"11111111111111111111111111111111","seq":3,"time":"2026-10-20T11:00:00Z","product":{"name":"bubo","version":"0.1.1","revision":"3ff74dce","engine_version":"0.10.0"},"type":"error","payload":{"code":"integrity_failed","component":"prepare","stage":"preflight","retryable":false,"remedy":"Remove the invalid file. Run prepare again.","detail":"different existing runtime file 'C:\\Users\\example\\AppData\\Local\\Bubo\\runtime\\rocm-10.0.0-gfx1151\\hipblas.dll'; delete it and run bubo prepare again"}}
```

```jsonl
{"schema_version":1,"operation_id":"11111111111111111111111111111111","seq":1,"time":"2026-10-13T18:00:00Z","product":{"name":"bubo","version":"0.1.0","revision":"3ff74dce","engine_version":"0.1.0"},"type":"error","payload":{"code":"invalid_arguments","component":"cli","stage":"test","retryable":false,"remedy":"Run the command again."}}
```

### Event `capabilities`

Required booleans: `prepare_jsonl`, `diagnose_json`, `status_json`, `instance_control`, and `status_watch`. `schema_version` is 1. The first three are true. Instance control is true on Windows and false on other platforms. Watch is true.

These booleans are a capability subset, not a complete command or platform
inventory. Additional advertised fields cover settings, memory estimates,
connection helpers, support bundles, switch registry, request history,
per-slot context and machine readings. Profiles are CLI-supplied rows. An
absent optional field is not proof that an otherwise documented command is
unsupported; use the command matrix and retain typed unsupported failures.

```jsonl
{"schema_version":1,"operation_id":"11111111111111111111111111111111","seq":1,"time":"2026-10-13T18:00:00Z","product":{"name":"bubo","version":"0.1.0","revision":"3ff74dce","engine_version":"0.1.0"},"type":"capabilities","payload":{"schema_version":1,"prepare_jsonl":true,"diagnose_json":true,"status_json":true,"instance_control":true,"status_watch":true}}
```

### Event `plan`

Required fields: `component`, `ready`, `sources`, `licenses`, `paths`, `volumes`, `plan_digest`, `record_kind`, `part`, and `parts`. `record_kind` is plan, source, license, path, or volume. Each non-header part has exactly one item of its named kind. Optional `file_states` lists download, reuse, import, or resume. For a model, a pinned file in the package folder that fails its pin (wrong size, fingerprint, or SHA-256) has the state `download` (`import` when the user supplied a good file for it), counts in `bytes_to_write` and in the download total, and the text plan adds "replaces a file that did not match" to its row. The model plan also carries `replaces`, an array of strings parallel to `sources`: the plain reason a file in the package folder is replaced (for example "SHA-256 differs from the pin"), or "" when nothing is replaced. It is in every part of the plan like `file_states`, and it is part of `plan_digest`, so approval of a plan that replaces a file does not carry over to a plan that does not. Source fields are `file_id`, `url`, `size`, `sha256`, `repository`, `revision`, `path`, and `license_id`. Licence fields are `path`, `text`, `sha256`, `text_offset`, and `text_bytes`. Volume fields are `name`, `free_bytes`, and `required_bytes`. Paths maps use string values. Hashes use 64 lowercase hexadecimal characters. Sizes and offsets use bytes.

```jsonl
{"schema_version":1,"operation_id":"11111111111111111111111111111111","seq":1,"time":"2026-10-13T18:00:00Z","product":{"name":"bubo","version":"0.1.0","revision":"3ff74dce","engine_version":"0.1.0"},"type":"plan","payload":{"component":"model","ready":false,"sources":[],"licenses":[],"paths":{},"volumes":[],"plan_digest":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","record_kind":"plan","part":0,"parts":5}}
```

Example component lines:

```jsonl
{"schema_version":1,"operation_id":"11111111111111111111111111111111","seq":1,"time":"2026-10-13T18:00:00Z","product":{"name":"bubo","version":"0.1.0","revision":"3ff74dce","engine_version":"0.1.0"},"type":"plan","payload":{"component":"model","ready":false,"sources":[{"file_id":"main","url":"https://example.invalid/pinned","size":4,"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","repository":"fixture","revision":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","path":"C:/test/model.gguf","license_id":"fixture"}],"licenses":[],"paths":{},"volumes":[],"plan_digest":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","record_kind":"source","part":1,"parts":5}}
{"schema_version":1,"operation_id":"11111111111111111111111111111111","seq":1,"time":"2026-10-13T18:00:00Z","product":{"name":"bubo","version":"0.1.0","revision":"3ff74dce","engine_version":"0.1.0"},"type":"plan","payload":{"component":"model","ready":false,"sources":[],"licenses":[{"path":"LICENSE","text":"Fixture license.","sha256":"bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb","text_offset":0,"text_bytes":16}],"paths":{},"volumes":[],"plan_digest":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","record_kind":"license","part":2,"parts":5}}
{"schema_version":1,"operation_id":"11111111111111111111111111111111","seq":1,"time":"2026-10-13T18:00:00Z","product":{"name":"bubo","version":"0.1.0","revision":"3ff74dce","engine_version":"0.1.0"},"type":"plan","payload":{"component":"model","ready":false,"sources":[],"licenses":[],"paths":{"model":"C:/test"},"volumes":[],"plan_digest":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","record_kind":"path","part":3,"parts":5}}
{"schema_version":1,"operation_id":"11111111111111111111111111111111","seq":1,"time":"2026-10-13T18:00:00Z","product":{"name":"bubo","version":"0.1.0","revision":"3ff74dce","engine_version":"0.1.0"},"type":"plan","payload":{"component":"model","ready":false,"sources":[],"licenses":[],"paths":{},"volumes":[{"name":"C:","free_bytes":100,"required_bytes":50}],"plan_digest":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","record_kind":"volume","part":4,"parts":5}}
```

### Event `preflight`

Required fields: `component` and `phase`. The phase is preflight. No approval or transfer is implied.

```jsonl
{"schema_version":1,"operation_id":"11111111111111111111111111111111","seq":1,"time":"2026-10-13T18:00:00Z","product":{"name":"bubo","version":"0.1.0","revision":"3ff74dce","engine_version":"0.1.0"},"type":"preflight","payload":{"component":"model","phase":"preflight"}}
```

### Event `approval_required`

Required fields: `plan_digest` and `components`. The component list names the plans bound by this approval. The digest is 64 lowercase hexadecimal characters.

```jsonl
{"schema_version":1,"operation_id":"11111111111111111111111111111111","seq":1,"time":"2026-10-13T18:00:00Z","product":{"name":"bubo","version":"0.1.0","revision":"3ff74dce","engine_version":"0.1.0"},"type":"approval_required","payload":{"plan_digest":"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc","components":["model"]}}
```

### Event `progress`

Required fields: `component`, `phase`, `file_id`, `bytes_done`, `bytes_total`, `resume_offset`, and `verification_method`. Phases are download, copy, verify, extract, or promote. `bytes_total` is null if unknown. Other byte values are nonnegative. `verification_method` identifies the check, normally sha256.

```jsonl
{"schema_version":1,"operation_id":"11111111111111111111111111111111","seq":1,"time":"2026-10-13T18:00:00Z","product":{"name":"bubo","version":"0.1.0","revision":"3ff74dce","engine_version":"0.1.0"},"type":"progress","payload":{"component":"model","phase":"download","file_id":"main","bytes_done":2,"bytes_total":4,"resume_offset":0,"verification_method":"sha256"}}
```

### Event `result`

Required fields: `exit_code`, `manifest_path`, and `prepared`. Success has exit_code 0 and prepared true. `manifest_path` is an absolute UTF-8 path, or null for runtime-only preparation. A failed operation ends with error instead of result when stdout remains usable.

```jsonl
{"schema_version":1,"operation_id":"11111111111111111111111111111111","seq":1,"time":"2026-10-13T18:00:00Z","product":{"name":"bubo","version":"0.1.0","revision":"3ff74dce","engine_version":"0.1.0"},"type":"result","payload":{"exit_code":0,"manifest_path":"C:/test/bubo-package.json","prepared":true}}
```

`bubo prepare` has no NPU side-jobs step in v0.1.1: the `--npu-sidejobs`
and `--accept-gemma-terms` options, exit code 11 and the `npu_sidejobs`
object in `result` are removed.

### Event `diagnose`

Required report fields: `schemaVersion` (1.0.0), `engineRevision`, `timestamp`, `status`, `checks`, `warnings`, `errors`, and `schema_version` (1). Status is PASS, WARN, or FAIL, as in the existing v1.0.0 report. Each check has `name`, `status`, `message`, and `details`. Check status also uses PASS, WARN, or FAIL. Details is an extension object. Warnings and errors are string arrays. A failed report requires a typed `error`. Existing report keys also remain at event top level. Additional inventory data keeps the existing diagnostic format.

```jsonl
{"schema_version":1,"operation_id":"11111111111111111111111111111111","seq":1,"time":"2026-10-13T18:00:00Z","product":{"name":"bubo","version":"0.1.0","revision":"3ff74dce","engine_version":"0.1.0"},"type":"diagnose","payload":{"schemaVersion":"1.0.0","engineRevision":"3ff74dce","timestamp":"2026-10-13T18:00:00Z","status":"PASS","checks":[{"name":"cpu","status":"PASS","message":"Available.","details":{}}],"warnings":[],"errors":[],"schema_version":1}}
```

### Event `diagnose_artifact`

Artifact extension object. Required: `schema_version` 1 and artifact data. Fingerprints contain `schemaVersion`, `fingerprintId`, and `canonical`. Validation contains `isValid`, `schemaVersion`, `fingerprintId`, `errors`, and `warnings`. Benchmark data keeps its existing artifact format. Optional `error` has the typed error shape. This event does not freeze the inner benchmark or inventory formats.

```jsonl
{"schema_version":1,"operation_id":"11111111111111111111111111111111","seq":1,"time":"2026-10-13T18:00:00Z","product":{"name":"bubo","version":"0.1.0","revision":"3ff74dce","engine_version":"0.1.0"},"type":"diagnose_artifact","payload":{"schema_version":1,"schemaVersion":"1.0.0","isValid":true,"fingerprintId":"","errors":[],"warnings":[]}}
```

### Event `status`

Required fields: `prepared`, `phase`, `versions`, `capabilities`, `paths`, `prepared_default`, `instances`, `effective_settings`, `memory`, `last_error`, `sample_time`, and `readiness`. Phase is not_prepared, stopped, starting, loading, ready, stopping, error or multiple_instances. Versions has the product fields. Paths has `runtime`, `local_app_data`, and `manifest`; `prepared_default_pointer` can be present. `runtime` is the AMD runtime folder in use or, if none is installed, the folder `bubo prepare runtime` will fill. It is chosen in this order: `BUBO_RUNTIME_DIR`, the shared folder `%LOCALAPPDATA%\Bubo\runtime\rocm-10.0.0-gfx1151` when its marker is valid, a valid runtime beside `bubo.exe`. `runtime_kind` is `shared`, `explicit` (from `BUBO_RUNTIME_DIR`) or `legacy` (beside `bubo.exe`), and `runtime_shared` is the shared folder of this user. Manifest, pointer, and prepared_default can be null. Readiness has boolean `runtime` and `model`. Optional `model_selection` has `model`, `manifest`, `pointer`, and `source`. Source is explicit_manifest, explicit_model, environment, prepared_default, or standard_location. Last_error is null or a typed error. Instances contains verified discovery descriptors. With several live instances, `selection_required` is true; select one explicitly. `identity`, `endpoint`, `log_path`, `last_transition`, `counters`, `last_completed_request`, `sessions`, `queue`, `slots` and `live_throughput` accompany each snapshot. Readiness keeps boolean runtime and model fields and adds `server` from the bound server predicate. Unavailable runtime fields are null. A missing model/runtime still returns valid status with exit 0. Readiness uses published metadata and file sizes; it is not full hash verification.

```jsonl
{"schema_version":1,"operation_id":"11111111111111111111111111111111","seq":1,"time":"2026-10-13T18:00:00Z","product":{"name":"bubo","version":"0.1.0","revision":"3ff74dce","engine_version":"0.1.0"},"type":"status","payload":{"prepared":false,"phase":"not_prepared","versions":{"name":"bubo","version":"0.1.0","revision":"3ff74dce","engine_version":"0.1.0"},"capabilities":{"schema_version":1,"prepare_jsonl":true,"diagnose_json":true,"status_json":true,"instance_control":true,"status_watch":true},"paths":{"runtime":"C:/Bubo","local_app_data":"C:/local","manifest":null},"prepared_default":null,"instances":[],"effective_settings":null,"memory":null,"last_error":{"code":"runtime_missing","component":"runtime","stage":"status","retryable":false,"remedy":"Run bubo prepare runtime."},"sample_time":"2026-10-13T18:00:00Z","readiness":{"runtime":false,"model":false},"identity":null,"endpoint":null,"log_path":null,"last_transition":null,"counters":null,"last_completed_request":null,"sessions":null,"queue":null,"slots":null,"selection_required":false,"live_throughput":{"label":"aggregate wall throughput","unit":"tokens_per_second","interval_ms":null,"prefill_tokens_per_second":null,"decode_tokens_per_second":null}}}
```

Live control reads a server-published snapshot; it does not call the scheduler.
`sample_time` is the start of the last completed runtime observation, not the
pipe response time. Optional `sample_age_ms` is its monotonic age at response.
A blocked observation keeps its previous time and increasing age. Machine
readings are published independently and keep their own observation times;
GPU power, clocks and local memory usage are read again on every machine pass.
Watch schedules records from the start of each iteration so response work does
not add a full extra second between records.

#### NPU prefill runtime

Status and ready payloads optionally include `npu_prefill_runtime`, the live
NPU state, independently of the startup value in `effective_settings`.
Builds without NPU support omit this object. A serve startup with NPU support
installs the producer publisher and publishes its current state once; a denied
startup publishes the existing authorization reason. The publisher updates
the published snapshot on state changes and at request end and is cleared on
shutdown, including startup failures. Status only copies that snapshot.

The object requires `state`, `reason`, `trips`, `hal_faults`, `whea_events`,
`invalid_chunks` and `chip_power`. State is on (armed), off (not armed, with
the existing startup reason), tripped (a timed gate hold for power or memory),
faulted (disabled for this process, reason fault), or unavailable (the runtime
cannot be queried). Reason is a string or null. The four counters are
nonnegative integers since server start. No gate currently emits tripped;
`trips` remains zero until that producer exists.

Chip power has its own `state`: ok, unavailable or off, plus required `w_mean`
(number or null) and `window_s` (number). Until the gate defines its window,
these describe the last request's NPU-active mean watts and active seconds.
Optional `memory` has required numeric `min_available_mib` and `max_commit_pct`
over the last request. Optional `scheme` has required string `guid` and `name`,
read once at NPU startup. Memory and scheme are omitted before observation;
reason and mean watts use null when absent. The object and its nested objects
reject unknown fields. This additive field keeps schema_version at 1.

### Event `starting`

Payload requires `identity`. Serve also has `instance_id`, `pid`, `process_creation_time` and `exe_path` at event top level.

```jsonl
{"schema_version":1,"operation_id":"11111111111111111111111111111111","seq":1,"time":"2026-10-13T18:00:00Z","product":{"name":"bubo","version":"0.1.0","revision":"3ff74dce","engine_version":"0.1.0"},"type":"starting","payload":{"identity":{"instance_id":"22222222222222222222222222222222","pid":1234,"process_creation_time":"134000000000000000","exe_path":"C:/Bubo/bubo.exe"}},"instance_id":"22222222222222222222222222222222","pid":1234,"process_creation_time":"134000000000000000","exe_path":"C:/Bubo/bubo.exe"}
```

### Event `loading`

Payload requires `identity`. Serve also has `instance_id`, `pid`, `process_creation_time` and `exe_path` at event top level.

```jsonl
{"schema_version":1,"operation_id":"11111111111111111111111111111111","seq":1,"time":"2026-10-13T18:00:00Z","product":{"name":"bubo","version":"0.1.0","revision":"3ff74dce","engine_version":"0.1.0"},"type":"loading","payload":{"identity":{"instance_id":"22222222222222222222222222222222","pid":1234,"process_creation_time":"134000000000000000","exe_path":"C:/Bubo/bubo.exe"}},"instance_id":"22222222222222222222222222222222","pid":1234,"process_creation_time":"134000000000000000","exe_path":"C:/Bubo/bubo.exe"}
```

### Event `ready`

Payload requires `identity`. Serve also has `instance_id`, `pid`, `process_creation_time` and `exe_path` at event top level.

```jsonl
{"schema_version":1,"operation_id":"11111111111111111111111111111111","seq":1,"time":"2026-10-13T18:00:00Z","product":{"name":"bubo","version":"0.1.0","revision":"3ff74dce","engine_version":"0.1.0"},"type":"ready","payload":{"identity":{"instance_id":"22222222222222222222222222222222","pid":1234,"process_creation_time":"134000000000000000","exe_path":"C:/Bubo/bubo.exe"},"endpoint":"http://127.0.0.1:49152/v1","served_model":"fixture","effective_settings":{"context":{"requested":4096,"effective":4096,"source":"cli","reason":null}},"log_path":null,"readiness":{"runtime":true,"model":true,"server":true}},"instance_id":"22222222222222222222222222222222","pid":1234,"process_creation_time":"134000000000000000","exe_path":"C:/Bubo/bubo.exe"}
```

### Event `warning`

Payload requires `identity`. Serve also has `instance_id`, `pid`, `process_creation_time` and `exe_path` at event top level.

```jsonl
{"schema_version":1,"operation_id":"11111111111111111111111111111111","seq":1,"time":"2026-10-13T18:00:00Z","product":{"name":"bubo","version":"0.1.0","revision":"3ff74dce","engine_version":"0.1.0"},"type":"warning","payload":{"identity":{"instance_id":"22222222222222222222222222222222","pid":1234,"process_creation_time":"134000000000000000","exe_path":"C:/Bubo/bubo.exe"},"code":"queue_pressure","remedy":"Reduce active GPU workloads."},"instance_id":"22222222222222222222222222222222","pid":1234,"process_creation_time":"134000000000000000","exe_path":"C:/Bubo/bubo.exe"}
```

### Event `stopping`

Payload requires `identity`. Serve also has `instance_id`, `pid`, `process_creation_time` and `exe_path` at event top level.

```jsonl
{"schema_version":1,"operation_id":"11111111111111111111111111111111","seq":1,"time":"2026-10-13T18:00:00Z","product":{"name":"bubo","version":"0.1.0","revision":"3ff74dce","engine_version":"0.1.0"},"type":"stopping","payload":{"identity":{"instance_id":"22222222222222222222222222222222","pid":1234,"process_creation_time":"134000000000000000","exe_path":"C:/Bubo/bubo.exe"}},"instance_id":"22222222222222222222222222222222","pid":1234,"process_creation_time":"134000000000000000","exe_path":"C:/Bubo/bubo.exe"}
```

### Event `stopped`

Payload requires `identity`. Serve also has `instance_id`, `pid`, `process_creation_time` and `exe_path` at event top level.

```jsonl
{"schema_version":1,"operation_id":"11111111111111111111111111111111","seq":1,"time":"2026-10-13T18:00:00Z","product":{"name":"bubo","version":"0.1.0","revision":"3ff74dce","engine_version":"0.1.0"},"type":"stopped","payload":{"identity":{"instance_id":"22222222222222222222222222222222","pid":1234,"process_creation_time":"134000000000000000","exe_path":"C:/Bubo/bubo.exe"}},"instance_id":"22222222222222222222222222222222","pid":1234,"process_creation_time":"134000000000000000","exe_path":"C:/Bubo/bubo.exe"}
```


## Local control and runtime status

Serve uses `\\.\pipe\Bubo.Control.<instance_id>`. Each request and response
uses `schema_version` 1 and `control_version` 1. Pipe requests use `action`
handshake, status, stop or received, with the exact `identity`. The response
echoes identity and action. Handshake requires `accepted` true. Status returns
`snapshot`. Stop returns `acknowledged` true and `stopped` false. The client
sends a received receipt before disconnect. Protocol errors return `exit_code`
and `code`. Records are UTF-8 JSON plus LF, with the common size limits.

The pipe rejects remote clients. Its protected DACL grants access only to the
current user SID. Discovery files and the instances directory use the same
owner-only rule. Atomic records under `%LOCALAPPDATA%\Bubo\instances` contain
identity, endpoint, phase, log_path and redacted last_error. Creation time is
the exact 64-bit Windows FILETIME represented by canonical decimal
text. The executable path is canonical. Discovery is a hint: status and stop
verify PID, creation time, canonical executable, process owner SID, pipe owner,
server PID and the versioned identity handshake. No process is selected by port.
Clean exit saves a redacted tombstone. Repeat stop succeeds only for an
owner-secured stopped record whose exact process has exited.

Serve takes an owner-only endpoint lease and reserves the socket before model
loading. It releases the reservation immediately before the authoritative HTTP
bind. Conflicts return port_occupied/13. The endpoint does not change silently.
Ready requires model/service readiness after bind. It has `served_model`,
`effective_settings`, `endpoint` and `log_path`. Each setting contains
`requested`, `effective`, `source` and `reason`. Non-echo environment entries
are excluded. Credentials are redacted. Null reason means no adjustment.
Serve reads the API key from `--api-key`, or from the environment variable
`BUBO_API_KEY` when the flag is absent. The flag wins if both are set and
differ. Use the variable to keep the key off the process command line. Serve
never prints the key.

Serve has a 64-record, 2 MiB output queue. Its worker alone writes stdout. A
write failure or queue limit permanently disables machine output. Inference,
control and atomic state continue. Human logs use stderr or the explicit file.
`--log-file PATH` selects the existing async sink. `--log-max-bytes N` defaults
to 8388608 bytes and accepts 4096 through 1073741824. `--log-retain N` defaults
to 3 backups and accepts 1 through 16. Both require log-file. Each file is
bounded by the byte limit. Status reports the absolute path, or null.

Stop emits `stopping` with acknowledged true and `process_exited` false. It
rejects new admission and cancels active work through the existing shutdown
path. It then waits on the retained process handle and emits `stopped` with
process_exited true. `already_stopped` marks a verified repeat. Exit 17 emits
stop_timeout and has no stopped completion. Stop never force-kills a process.
There is no HTTP shutdown endpoint.

Live rates use token-counter deltas over monotonic client sample time. Their
label is "aggregate wall throughput" and unit is tokens_per_second. The first
sample, a changed instance and a reset counter have null rates. Last completed
request gauges have age_ms; missing durations are null. Cache disk, memory and
miss totals increment once at the existing successful completion hook, with
disk taking precedence over memory. Cancelled and failed requests do not add
cache completions. RSS and host-available memory are independent byte values.
Sessions counts active requests, queue counts deferred requests, and slots
contains the existing backend session states; unsupported slots are null.

Runtime fields: `acknowledged`, `age_ms`, `age_unit`, `already_stopped`, `cache_disk_hits_total`, `cache_hit_ratio`, `cache_memory_hits_total`, `cache_misses_total`, `cached_prompt_tokens`, `cached_tokens_total`, `context_tokens`, `control_version`, `decode_seconds_total`, `decode_tokens_per_second`, `decode_tokens_total`, `device_lost_total`, `draft_accepted_tokens_total`, `draft_rounds_total`, `draft_tokens_total`, `effective`, `generated_tokens`, `host_available_bytes`, `id`, `interval_ms`, `label`, `max_sequence_tokens`, `prefill_seconds_total`, `prefill_tokens_per_second`, `prefill_tokens_total`, `prefilled_tokens`, `process_exited`, `processing`, `prompt_tokens`, `rate_unit`, `reason`, `remaining_tokens`, `request_id`, `requested`, `rss_bytes`, `served_model`, `speculative`, `unit`.

The fake backend is compiled only into the isolated CPU target with
`BUBO_SERVE_TESTING`. It uses the production command wrapper, lifecycle,
control, log sink and HTTP server. Package scanning rejects its flag strings
and symbols. Production has no fake-backend flag.

## Known issues: serve and control

Machine `serve video --jsonl` returns unsupported/18. The current video job
service validates its inventory at start and loads weights on demand. It
cannot certify loaded weights for a ready lifecycle event. Human video serving
retains its existing behavior. This is an explicit lifecycle limitation.

The CPU fake validates the command, transport and control paths. It does not
validate a real model load, device initialization or inference. These require
the separate manual validation.

## Error codes

Exit values are command-scoped. Lifecycle values apply to machine serve and control.
The error `unsupported` uses 18 for status, serve, and stop requests; other
unsupported machine commands use 2. Prepare's reserved value is 10.

| Code | Exit | Meaning and action |
|---|---|---|
| `unexpected_failure` | 1; bootstrap 2 | Unexpected failure. Read the remedy. |
| `invalid_arguments` | 2 | Invalid command or input. Correct the fields. |
| `cancelled` | 3 | Declined or cancelled. Run Prepare when needed. |
| `source_missing` | 4 | Pinned or offline source is missing. Restore the approved files. |
| `transport_failed` | 5 | Transport or TLS failed. Check the connection. |
| `integrity_failed` | 6 | A hash or identity failed, or an existing file differs from its pin (`detail` names it). A model file in the package folder is repaired by prepare instead (see the model plan); for any other file, replace the invalid file. |
| `storage_failed` | 7 | Storage or access failed. Check space and access. |
| `prepare_locked` | 8 | Another Prepare has the lock. Wait for it. |
| `prepare_internal` | 9 | Prepare archive or internal failure. Run Prepare again; read `detail` if it fails again. |
| `unsupported` | 18; other commands 2; reserved Prepare 10 | The requested capability is not implemented. Check capabilities. |
| `runtime_missing` | 11 lifecycle; Prepare prerequisite 4; Diagnose bootstrap 2 | Install the required runtime or repair the package. |
| `model_missing` | 12 lifecycle | Model or manifest is missing. Prepare the model. Status can include this in last_error with exit 0. |
| `port_occupied` | 13 lifecycle | Select another port. |
| `startup_timeout` | 14 lifecycle | Startup did not finish before the deadline. |
| `protocol_mismatch` | 15 lifecycle; Diagnose bootstrap 2 | Versions do not agree. Reinstall one complete package. |
| `control_access` | 16 lifecycle | Control identity or access failed. |
| `stop_timeout` | 17 lifecycle | Graceful stop deadline expired. Do not report stopped. |
| `memory_insufficient` | 19 lifecycle | The sessions at this context do not fit the machine even with the free dedicated GPU memory plus all available RAM (the keep-free floor is ignored here: a configuration that runs thin on RAM still starts). The error carries `needed_bytes`, `available_bytes`, `sessions`, `context`, `per_session_bytes`, `floor_bytes` (the keep-free floor in force, 0 when off) and `detail` (the same text for a human). Lower `--context` or `--sessions`, or free memory. A start that fits only with part of the session state in system RAM, or that eats into the keep-free floor, is not an error: the log carries `event=memory_guard action=load_spill` with the numbers and the remedy (fewer sessions, a smaller context, or a lower `GUFO_CACHE_HOST_FLOOR_MB`). |
| `device_lost` | 75 machine lifecycle | The device was lost. Retain the raw exit. |
| `diagnostic_failed` | 1 | A diagnostic check or validation failed. Read its findings. |

One example event line for each error code:

```jsonl
{"schema_version":1,"operation_id":"11111111111111111111111111111111","seq":1,"time":"2026-10-13T18:00:00Z","product":{"name":"bubo","version":"0.1.0","revision":"3ff74dce","engine_version":"0.1.0"},"type":"error","payload":{"code":"unexpected_failure","component":"cli","stage":"test","retryable":false,"remedy":"Run the command again."}}
{"schema_version":1,"operation_id":"11111111111111111111111111111111","seq":1,"time":"2026-10-13T18:00:00Z","product":{"name":"bubo","version":"0.1.0","revision":"3ff74dce","engine_version":"0.1.0"},"type":"error","payload":{"code":"invalid_arguments","component":"cli","stage":"test","retryable":false,"remedy":"Run the command again."}}
{"schema_version":1,"operation_id":"11111111111111111111111111111111","seq":1,"time":"2026-10-13T18:00:00Z","product":{"name":"bubo","version":"0.1.0","revision":"3ff74dce","engine_version":"0.1.0"},"type":"error","payload":{"code":"cancelled","component":"cli","stage":"test","retryable":false,"remedy":"Run the command again."}}
{"schema_version":1,"operation_id":"11111111111111111111111111111111","seq":1,"time":"2026-10-13T18:00:00Z","product":{"name":"bubo","version":"0.1.0","revision":"3ff74dce","engine_version":"0.1.0"},"type":"error","payload":{"code":"source_missing","component":"cli","stage":"test","retryable":false,"remedy":"Run the command again."}}
{"schema_version":1,"operation_id":"11111111111111111111111111111111","seq":1,"time":"2026-10-13T18:00:00Z","product":{"name":"bubo","version":"0.1.0","revision":"3ff74dce","engine_version":"0.1.0"},"type":"error","payload":{"code":"transport_failed","component":"cli","stage":"test","retryable":false,"remedy":"Run the command again."}}
{"schema_version":1,"operation_id":"11111111111111111111111111111111","seq":1,"time":"2026-10-13T18:00:00Z","product":{"name":"bubo","version":"0.1.0","revision":"3ff74dce","engine_version":"0.1.0"},"type":"error","payload":{"code":"integrity_failed","component":"cli","stage":"test","retryable":false,"remedy":"Run the command again."}}
{"schema_version":1,"operation_id":"11111111111111111111111111111111","seq":1,"time":"2026-10-13T18:00:00Z","product":{"name":"bubo","version":"0.1.0","revision":"3ff74dce","engine_version":"0.1.0"},"type":"error","payload":{"code":"storage_failed","component":"cli","stage":"test","retryable":false,"remedy":"Run the command again."}}
{"schema_version":1,"operation_id":"11111111111111111111111111111111","seq":1,"time":"2026-10-13T18:00:00Z","product":{"name":"bubo","version":"0.1.0","revision":"3ff74dce","engine_version":"0.1.0"},"type":"error","payload":{"code":"prepare_locked","component":"cli","stage":"test","retryable":false,"remedy":"Run the command again."}}
{"schema_version":1,"operation_id":"11111111111111111111111111111111","seq":1,"time":"2026-10-13T18:00:00Z","product":{"name":"bubo","version":"0.1.0","revision":"3ff74dce","engine_version":"0.1.0"},"type":"error","payload":{"code":"prepare_internal","component":"cli","stage":"test","retryable":false,"remedy":"Run the command again."}}
{"schema_version":1,"operation_id":"11111111111111111111111111111111","seq":1,"time":"2026-10-13T18:00:00Z","product":{"name":"bubo","version":"0.1.0","revision":"3ff74dce","engine_version":"0.1.0"},"type":"error","payload":{"code":"unsupported","component":"cli","stage":"test","retryable":false,"remedy":"Run the command again."}}
{"schema_version":1,"operation_id":"11111111111111111111111111111111","seq":1,"time":"2026-10-13T18:00:00Z","product":{"name":"bubo","version":"0.1.0","revision":"3ff74dce","engine_version":"0.1.0"},"type":"error","payload":{"code":"runtime_missing","component":"cli","stage":"test","retryable":false,"remedy":"Run the command again."}}
{"schema_version":1,"operation_id":"11111111111111111111111111111111","seq":1,"time":"2026-10-13T18:00:00Z","product":{"name":"bubo","version":"0.1.0","revision":"3ff74dce","engine_version":"0.1.0"},"type":"error","payload":{"code":"model_missing","component":"cli","stage":"test","retryable":false,"remedy":"Run the command again."}}
{"schema_version":1,"operation_id":"11111111111111111111111111111111","seq":1,"time":"2026-10-13T18:00:00Z","product":{"name":"bubo","version":"0.1.0","revision":"3ff74dce","engine_version":"0.1.0"},"type":"error","payload":{"code":"port_occupied","component":"cli","stage":"test","retryable":false,"remedy":"Run the command again."}}
{"schema_version":1,"operation_id":"11111111111111111111111111111111","seq":1,"time":"2026-10-13T18:00:00Z","product":{"name":"bubo","version":"0.1.0","revision":"3ff74dce","engine_version":"0.1.0"},"type":"error","payload":{"code":"startup_timeout","component":"cli","stage":"test","retryable":false,"remedy":"Run the command again."}}
{"schema_version":1,"operation_id":"11111111111111111111111111111111","seq":1,"time":"2026-10-13T18:00:00Z","product":{"name":"bubo","version":"0.1.0","revision":"3ff74dce","engine_version":"0.1.0"},"type":"error","payload":{"code":"protocol_mismatch","component":"cli","stage":"test","retryable":false,"remedy":"Run the command again."}}
{"schema_version":1,"operation_id":"11111111111111111111111111111111","seq":1,"time":"2026-10-13T18:00:00Z","product":{"name":"bubo","version":"0.1.0","revision":"3ff74dce","engine_version":"0.1.0"},"type":"error","payload":{"code":"control_access","component":"cli","stage":"test","retryable":false,"remedy":"Run the command again."}}
{"schema_version":1,"operation_id":"11111111111111111111111111111111","seq":1,"time":"2026-10-13T18:00:00Z","product":{"name":"bubo","version":"0.1.0","revision":"3ff74dce","engine_version":"0.1.0"},"type":"error","payload":{"code":"stop_timeout","component":"cli","stage":"test","retryable":false,"remedy":"Run the command again."}}
{"schema_version":1,"operation_id":"11111111111111111111111111111111","seq":1,"time":"2026-10-13T18:00:00Z","product":{"name":"bubo","version":"0.1.0","revision":"3ff74dce","engine_version":"0.1.0"},"type":"error","payload":{"code":"device_lost","component":"cli","stage":"test","retryable":false,"remedy":"Run the command again."}}
{"schema_version":1,"operation_id":"11111111111111111111111111111111","seq":1,"time":"2026-10-13T18:00:00Z","product":{"name":"bubo","version":"0.1.0","revision":"3ff74dce","engine_version":"0.1.0"},"type":"error","payload":{"code":"diagnostic_failed","component":"cli","stage":"test","retryable":false,"remedy":"Run the command again."}}
{"schema_version":1,"operation_id":"11111111111111111111111111111111","seq":1,"time":"2026-10-13T18:00:00Z","product":{"name":"bubo","version":"0.1.0","revision":"3ff74dce","engine_version":"0.1.0"},"type":"error","payload":{"code":"memory_insufficient","component":"serve","stage":"startup","retryable":false,"remedy":"Lower --context or --sessions, or free memory.","detail":"2 sessions do not fit: each needs 62.0 GiB of session state at this context (124.0 GiB in all), and the machine has 81.0 GiB free after loading the model. Lower --context or --sessions.","needed_bytes":133143986176,"available_bytes":86973087744,"sessions":2,"context":262144,"per_session_bytes":66571993088,"floor_bytes":4294967296}}
```

## Exit codes

| Exit | Meaning | Example command or condition |
|---|---|---|
| 0 | Success, help, or valid stopped status. Diagnose pass and warn. | `bubo status --json` |
| 1 | Diagnose failed check or unexpected status failure. Lifecycle unexpected failure. | `bubo diagnose --validate-artifact missing.json --json` |
| 2 | Invalid arguments or input. Diagnose engine load errors keep this legacy value. | `bubo prepare --bad --jsonl` |
| 3 | Prepare declined, cancelled, or consumer failed. | Send the cancel input line. |
| 4 | Prepare missing source or prerequisite. | `bubo prepare model --offline --jsonl` with missing files. |
| 5 | Prepare transport/TLS failure. | Transport reports a failed request. |
| 6 | Prepare integrity failure. | A pinned hash does not agree. |
| 7 | Prepare storage/access failure. | Required space exceeds free space. |
| 8 | Prepare lock is in use. | Two prepares try to mutate the same destination. |
| 9 | Prepare internal/archive failure. | An archive check fails. |
| 10 | Existing Prepare NotImplemented value. Reserved. | No successful operation uses this code. |
| 11 | Machine runtime prerequisite failure. | Serve has no runtime. |
| 12 | Machine missing model/manifest. | Serve has no manifest. |
| 13 | Machine occupied port. | Serve cannot bind its port. |
| 14 | Machine startup deadline. | Serve watchdog expires. |
| 15 | Machine protocol mismatch. | Control versions do not agree. |
| 16 | Machine identity/access failure. | Control rejects an instance ID. |
| 17 | Machine graceful stop deadline. | Stop deadline expires. |
| 18 | Unsupported machine lifecycle/control capability. | `bubo serve video --jsonl` |
| 19 | Machine memory refusal: the sessions at this context do not fit the machine. | `bubo serve --sessions 4 --context 262144 --jsonl` on a machine without that much free memory. |
| 75 | Existing device-lost value, retained. | A device-lost process exit. |

## Prepared selection

Prepare publishes the manifest before saving
`%LOCALAPPDATA%\Bubo\prepared-default.json`. The pointer also records a custom
output folder. The shared selection order is explicit manifest or model,
`BUBO_MANIFEST`, saved prepared-default pointer, then the standard package
location under `%LOCALAPPDATA%\Bubo\models`. Explicit manifest takes priority
over explicit model. A damaged saved pointer fails selection. It does not
cause a disk search or a silent fallback. Status reports the selection and
its source. Running instance selection uses verified local control.

`bubo status --json --manifest PATH` (a `bubo-package.json`, or its folder) and
`--model PATH` (a model file or folder; the marker beside it) take the same
selection as serve, with the same priority. Status then answers for that
package: `readiness.model` is true when the package is ready, `prepared_models`
has its row, `paths.manifest` and `model_selection` name it. A catalogue
package is ready when every file of its entry is present at its pinned size
(the size check of `models list`); the saved default (Flash Next) keeps its
fuller check. Without the flags nothing changes. Giving both flags is an
`invalid_arguments` error.

## Known issues: start, status and prepare

Windows can fail before `main` if a bootstrap import is missing. The CLI
cannot emit JSON, a capability header, a typed error, or a remedy in that
case. Keep the raw launch error or process status, such as NTSTATUS
`0xC0000135` for a missing imported DLL. Stdout can be empty. Do not infer a
CLI error code from that native value. A failed engine `LoadLibraryExW`
after bootstrap entry can emit `runtime_missing`. Status and Prepare route
before the engine load and do not require the AMD engine runtime to start.
The post-entry Prepare VC++ check returns runtime_missing with exit 4; it
does not demonstrate that the OS can start an executable with no CRT.

Status and registry listing run before engine load. A stopped
server has null runtime settings, inference defaults, slots and requests.
Memory estimates support only model packages that have a coefficient table.
Today that is the compiled shipped model package. Other models return null
with `memory_estimate_reason`. Stopped-server status has no estimate; the
estimate and its `recommended_context` appear in `serve --print-effective-settings`
output and in live status and ready.

## Contract files and checks

- Public text: `docs/CLI-CONTRACT.md`.
- Output schema: `tests/cli/contract/event.schema.json`.
- Input schema: `tests/cli/contract/input.schema.json`.
- Examples: `tests/cli/contract/valid/` and `input-valid/`.
- Rejected examples: `tests/cli/contract/invalid/` and `input-invalid/`.
- Check documentation: `python tests/cli/contract/check_docs.py --self-test`.
- Check schema fixtures: `pwsh -NoProfile -File tests/cli/contract/schema_test.ps1`.
- Run compiled CPU proof: `ctest --test-dir BUILD -L cpu --output-on-failure`.

The docs checker rejects event or error names missing from either the docs
or schema. It also requires examples and field names. A static check is not
proof that an executable emitted valid records. Run the native process and
fault tests on the candidate, then the production package/source gates.
## Settings, memory and connection

Prepare plan volume rows optionally include `bytes_to_write`,
`safety_margin_bytes`, and `free_after_download_bytes`. All are integer bytes,
with the same volume observation as `free_bytes`. `required_bytes` includes
the safety margin; `bytes_to_write` excludes it. Remaining free space subtracts
only planned writes from observed free space, clamped at zero. These writes
include planned metadata, notices, copies and extraction, where applicable.
They exclude reused bytes. The margin is not consumed by a download.
The model and runtime planners share this calculation. Split volume records
preserve these fields. Free-space observations and their derived remaining
space do not change the approval digest.

Live status, ready and each selected instance summary optionally include
`active_model`: nullable `catalogue_id`, `model_path`, and `manifest_path`.
Paths identify the selected files as absolute paths. A verified package supplies
its catalogue ID, independent of the served display name or an API name override.
An explicit file without known catalogue provenance retains a null ID; clients
can estimate by its path. Without an active selection, this field is null. These fields are additive
in schema_version 1 and do not replace `served_model`.

Live text status and ready optionally include `client_snippets`, an array of
structured templates. Each has `kind`, `title`, `template`, `requires_key`,
and nullable `default_api_key`. The CLI supplies OpenAI Python SDK, curl OpenAI
chat, PowerShell Invoke-RestMethod, and Anthropic Python SDK examples. Templates
contain `{openai_base_url}` or `{anthropic_base_url}` and `{api_key}` placeholders;
these are literal substitution tokens, not a general format-string language.
Other braces in the examples remain literal. The GUI selects a reachable address
from the bound `endpoints` and the user's selected LAN address, then substitutes
only those tokens. OpenAI URLs include /v1; Anthropic uses the origin.
`api_key_env` supplies the credential environment name, never its value.
When `requires_key` is false, `default_api_key` supplies the same loopback dummy
key as `pi_snippet`. When true it is null. Secrets are never filled into emitted
templates. Before binding and for non-text services, templates are null.
Wildcard and LAN bindings retain usable templates. Existing `pi_snippet`
behavior, including its loopback-only restriction, remains unchanged.

Live status and ready include `inference_defaults`, `effective_switches`,
`memory_estimate`, `memory_estimate_reason`, `pi_snippet`, `requests`,
`prepared_models`, and the reasoning controls. Null means unavailable.
Additions to version 1 remain optional for old output fixtures. When an object
is present, its defined required fields must be present. Clients ignore unknown
keys at every output object, and reject missing required fields. Prepare input
remains strict. No existing exit code was changed.

`effective_settings` retains the setting rows. Each setting has `requested`,
`effective`, `source`, `reason`, and `allowed`. The CLI resolver supplies profile
rules. Clients must not repeat them. Resolver origins are retained in `origin`;
the public switch source is default, profile, user or env. Manifest values use
profile as the public source and manifest as the detailed origin. Environment
overrides that affect the result appear in `environment_overrides`. Values that
are not echo-safe and authentication credentials are omitted or redacted.

Settings `capabilities` supplies profile name, label and `gui` selection. Offer
Default, Exact and Vanilla. Subagents is CLI only. Exact and Vanilla force NPU
prefill off. A build without NPU support also forces it off. Startup failure
reports the actual NPU fallback and reason. An explicit `--host 127.0.0.1` and
chosen `--port` take precedence over HOST, PORT, GUFO_HOST and GUFO_PORT.

`inference_defaults` has temperature, top_p, top_k, min_p, repeat_penalty,
presence_penalty, seed, reasoning_effort, think and max_tokens. Every entry
contains a typed `value` and `source` (default, flag, profile or env). A zero
max_tokens value is the backend's no-explicit-output-limit default. It is not
a thinking limit. The server reports the model's resolved sampling values.

`--think-marker-bias NUMBER` sets the finite bias through the resolver. Its
explicit value overrides the environment bias and per-effort map.
Its registry `allowed` minimum is -2 and maximum is 0, the documented
low:-2, medium:-1, xhigh:0 effort-map range for GUI input. There is no step.
The engine continues to accept any finite number; these presentation bounds
do not add engine clamping or a new parser restriction.
`--loop-breaker on|off` sets the loop breaker through the resolver. Both retain
the Exact profile constraints. `think_marker_bias` contains value, source,
requested and `by_effort` entries for low, medium, high and xhigh. The high
entry follows the existing high-to-xhigh resolver mapping. Each per-effort
entry contains value and source. `loop_breaker` contains a boolean value,
source and requested value. `think_marker_entropy_gate` is `{available:false}`
until R7. No new thinking-token budget, cap or force-close control is defined.

`slots` uses the backend's session states. Each slot includes id, state
(idle, prefilling or generating), context_tokens, tokens_used and request_id.
tokens_used is active processed plus cached prompt tokens plus generated tokens;
idle slots follow the backend's zero-count convention. This is not a retained
cache occupancy measure. `capabilities.per_slot_context` is false. All slots
use the same configured context. Unsupported slot state is null.

`requests` holds at most 50 generation request completions in completion order.
Each row contains id, time, slot, prompt_tokens, completion_tokens, ttft_ms,
prefill_tps, decode_tps, cached_tokens, cache_kind, finish_reason, outcome,
status and error_code. Rates are tokens per second; ttft_ms is milliseconds.
Missing Result timing or slot data is null. Cache kind is disk, memory or miss.
Finish reason is stop, stop_sequence, length or cancelled. Outcome is completed,
disconnected, stream_error, failed or refused. `status` is the HTTP status of the
reply (null when none was sent) and `error_code` the stable error code of a failed
reply or a stopped stream (null otherwise). Outcome refused marks the three
memory-guard codes `insufficient_memory` (503, retry later),
`insufficient_memory_permanent` (507, the machine can never provide it) and
`memory_pressure` (503, a running request was stopped because Windows ran low on
memory); such a row also carries `refusal` with `code`, `needed_bytes` and
`free_bytes` (the two numbers are null for a stopped request). Failure rows retain
null Result fields when no Result is available. No prompt, response text, client address or request body is kept.
`request_completed` emits the same row through serve JSONL at quiet log level.
It does not change lifecycle phase or last_transition.

`prepared_models` has one entry for the compiled package: id, display_name,
manifest_path, size_bytes, prepared_state and capabilities (context_max, mtp,
twin, mmproj). The size is the pinned package download size, not a memory
estimate. `prepared_state` uses existing publication and file checks. Serve
retains the full verification path.

The bound `endpoint` includes /v1, and `served_model` is the backend's actual
model id. `pi_snippet` is a models.json provider fragment. It has baseUrl,
`api: openai-completions`, apiKey, and models with id, name and contextWindow.
Unauthenticated loopback uses the dummy key bubo-loopback. Authenticated
loopback uses BUBO_API_KEY as an environment-variable reference, never the
actual key. The helper is unavailable for a non-loopback endpoint or unknown
text context. The CLI never writes a Pi file. See pi-snippet.schema.json.

### Memory coefficient table, version 1

The CLI owns `MemoryCoefficientsV1` in src/cli/settings_support.hpp. Every
amount carries raw bytes, gib and evidence. A GiB is 1073741824 bytes; a
decimal GB is 1000000000 bytes; a MiB is 1048576 bytes. Budgets are null until
an actual budget observation is supplied. No 64 GiB fit is claimed.

| Coefficient | Bytes or formula | Evidence and conditions |
| --- | --- | --- |
| Main weights | 82500000000 | Estimated from the recipe's rounded 77.0 + 5.5 decimal GB weight breakdown. |
| MTP / twin / draft head / vision | 2790000000 / 1230000000 / 360000000 / 910000000 | Estimated rounded recipe coefficients, included only when selected. |
| Reference executor state | 7176 * 1048576 = 7524581376 | Measured aggregate executor coefficient at 262144 context tokens, one session. Vision state is separate. |
| Fixed session portion | 259 * 1048576 | Estimated split; it is not a separate measurement. |
| Context portion | (7176 - 259) * 1048576 * context / 262144, rounded up | Estimated scaling, independent of occupied tokens. |
| Scratch/phase | 3460000000 (min and max) | Decode-approved runtime overhead plus deferred scratch; charged once per model. |
| Per-session vision | 12 * context + 10240 * min(context, image cap) + 64 | Flash Next positions, 2560-wide float embeddings and control. Zero without a bound vision encoder. GUFO_MAX_IMAGE_TOKENS supplies the cap; unset uses 65536 (counting every image in the prompt), still limited by context. |
| Device cache/snapshot | Additional allocated snapshot pool allowance | Flash Next retention limits (including the explicit profile limit) are ceilings, not startup allocations. Borrowed snapshots share live state until copy-on-write; no unconditional full-session mirror is added. On-demand growth is excluded from startup admission. |
| Host process | 0.85 * 1073741824, rounded down | Estimated base process RAM. |
| Host NPU staging | 3.5 * 1073741824 when NPU prefill is enabled | Estimated host staging. |
| Host cache | Selected cache limit for A3B/Gemma host vectors | Other device allocations stay in the GPU total. Unsupported model estimates remain null. |
| Device reserve | 4 * 1073741824 | Snapshot-pool sizing only; excluded from GPU need and the admission verdict. |
| Host floor | 4 * 1073741824 | Planning minimum free host RAM. The engine snapshot-admission built-in is 4096 MiB; a null registry profile value means no injection. |

Source: the measured release recipe table and the placement correction of
2026-10-14.
Weight, executor-state and reserve coefficients remain independent of catalogue file sizes. Only scratch was recalibrated with decode approval.
The aggregate state coefficient has measured evidence; the complete estimate is
estimated and `fit_verified` stays false. `total_range` is the GPU requirement
excluding the snapshot-pool reserve. `per_session_state` is executor-only;
`per_session_vision` and `vision_session_total` expose vision separately.
`session_state_total` includes both executor and vision. `host_cache_snapshot_allowance` reports only
host cache storage; `device_cache_snapshot_allowance` reports device cache storage.

Additive optional version-1 fields `gpu {available, total, dedicated_video_memory,
need, need_max, reserve}` and `host {available, total, need, floor}` contain
memory amounts (bytes, gib, evidence). The `total` reading is independent
of available memory. `gpu.total` is the measured DXGI LOCAL
plus NON_LOCAL Budget, before whole-GiB flooring by the shared engine cache
planner; it is null when either segment cannot be read. `host.total` is measured
physical host RAM. `gpu.dedicated_video_memory` is the measured adapter carve,
read from DXGI IDXGIAdapter::GetDesc1 DedicatedVideoMemory on the same selected
AMD adapter as the budget. `dedicated_video_memory_source` names that API;
`dedicated_video_memory_reason` is null when available, otherwise states why
the field is null. A missing/zero reading never becomes a zero carve.
`available`, `need`, `need_max`, `floor` and `reserve` are typed memory amounts.
`need` uses the lower scratch estimate; `need_max` retains the upper estimate.
`cache_mode` distinguishes `automatic` from `explicit`; `fit_scope` is
`conservative_startup_estimate` for Flash Next on-demand cache growth (including explicit retention ceilings),
`startup_lower_bound` for other automatic estimates, and `planning` otherwise.
Startup estimates explicitly report that on-demand cache and snapshot growth are not included;
it never treats zero as disabled. `fits` is true only when GPU need and host need plus floor both fit;
it is null if either budget is unknown. `fit_reason` names the short side or
missing budget. This remains a planning lower bound, including load-dependent
cache/snapshot growth only when its allowance is supplied.

The read-only Windows probe reuses the machine adapter's DXGI reader. GPU
availability is `DXGI local Budget - CurrentUsage`,
clamped to zero and sampled before loading. Shared system memory is never added.
Missing budget/usage readings keep the budget unknown. DedicatedVideoMemory does
not cap the local budget that HIP can allocate. Host availability
comes from `GlobalMemoryStatusEx.ullAvailPhys` and is independent of the GPU carve.

### Recommended context per session count

`memory_estimate.recommended_context` uses the same estimator and placement for
all session counts, from 1 through the model's `max_sessions`. `available_bytes`
and `available_gib` now describe the GPU carve available before loading.
`range_basis` is `gpu.need`; `margin_bytes` and `margin_gib` report the
recommendation margin max(5 GiB, 5% of available GPU memory).
Old version-1 range bases remain valid. The top-level
`recommended` and `recommendation_reason` describe the requested context and
session count independently of `fits` and `fit_reason`.
For Flash Next lazy retention, startup admission (`fits`) stays unchanged.
Recommendation additionally requires:

- R2: GPU available minus startup need minus cache-at-clamp must be at least
  max(5 GiB, 5% of GPU available). Cache-at-clamp comes from the same
  `PlanSnapshotCapacity` function used by the runner: host total minus resident
  host bytes and reserve, limited by whole-GiB device total minus weights,
  all per-session executor and vision claims, and reserve. The explicit profile
  limit clamps to that ceiling; automatic mode uses the engine's half-ceiling
  rule and maximum. These are limits on lazy growth, not startup allocations.
- R3: spill is max(0, startup GPU need + cache-at-clamp - dedicated video memory).
  Host available minus the process/NPU/cache host need minus spill must retain
  the host floor. Vision remains included in the GPU need. Spill is a
  recommendation term only and never changes admission.

The selected setup and every context row add nullable `cache_at_clamp_bytes`,
`host_spill_bytes`, `free_after_growth_gib` and `host_free_after_spill_gib`.
`free_after_gib` remains the row's startup headroom. Reasons name cache growth,
remaining GPU margin and host spill/floor, including successful verdicts. If
DedicatedVideoMemory cannot be read, spill and host-free-after-spill are null;
the recommendation remains computable using R2 and explicitly states that
host spill was not checked. Missing totals prevent confirming lazy growth,
with a reason and an unconfirmed recommendation. No capacity is silently
substituted from free memory.
The CLI publishes both verdicts; GUI readers do not recalculate them.

The selected estimate and every `recommended_context.by_sessions` row emit
`recommendation_failures`, an array containing `gpu_headroom` when R2 fails,
`host_floor` when R3 fails, or both when both checks fail. An admission shortfall
already establishes the corresponding failed term even before cache growth or
spill is known. `unconfirmed` means a measurement required by the existing
recommendation path is missing (GPU/host availability or lazy-growth totals).
Confirmed failures can accompany `unconfirmed`. The array is empty when
`recommended` is true. A missing dedicated carve alone preserves the existing
R2-only verdict and does not add `unconfirmed`; the reason still says spill was
not checked. These tags come from the same evaluator as the verdict and do not
change `recommended` or `recommendation_reason`. The field is optional in the
version-1 schema to accept older records; current producers always include it
on non-null estimates and every context row. Readers must treat an absent field
as unavailable detail, without deriving tags from the reason text or formulas.

Each row lists `sessions`, `context_per_session`, `fits`, `total_min_bytes`,
`free_after_gib`, optional `recommended` and optional `reason`. A fallback minimum
context can fit admission while lacking the recommendation margin; automatic
context selection requires the row to be recommended. The recommendation fields `available_evidence`, `margin_rule`, `step_tokens`,
`minimum_context_tokens`, `context_max_tokens` and `by_sessions` describe the
observed budget, stepping rule and rows. `recommended_context_reason` explains
an unavailable recommendation. The separate `host` budget never becomes GPU RAM.

The CLI chooses the largest context in
steps of 8192, capped at the model's own limit, for which admission, cache-growth margin and the available host-spill check pass.
If none is recommended it shows the minimum context with `recommended: false`;
`fits` independently retains the admission verdict and the reason names the
short or unknown side. More sessions never give a larger context. Missing
GPU availability produces a null recommendation with a reason; missing host
availability keeps rows unconfirmed. No UI memory formula is needed.

`connection` contains the bound endpoint, model_id, context_tokens and
authenticated boolean. The listener supplies the bound address and assigned
port. Missing connection data is null. IPv6 addresses use brackets in the URL.

`endpoints` is optional on status and ready. Before binding it is null.
For a bound listener it contains
`openai_base_url` (the bound origin plus `/v1`) and `anthropic_base_url` (the bound
origin). OpenAI SDKs append `/chat/completions`; Anthropic SDKs append `/v1/messages`.
The actual bound host and assigned port supply both URLs, including IPv6 brackets.
`api_key_env` contains only the name `BUBO_API_KEY`, never its value. A not-running
status has null endpoints and a null memory estimate.

`prepare --output ROOT` changes model and archive-cache storage. Runtime selection
uses the same resolver as serve/status: explicit `--runtime-dir`, `BUBO_RUNTIME_DIR`,
then the shared per-user location or an existing usable legacy runtime beside the
executable. A custom output root does not create an undiscoverable runtime copy.

## Models and the model catalogue

`bubo models list` and `bubo models estimate` tell a client what it can run and
how much memory it needs, before any server starts. The GUI computes none of
this. Both commands read file headers only. They never start a server, load
weights, use the GPU or use the network.

```text
bubo models list [--json|--jsonl] [--scan-folder PATH]... [--models-root PATH]
bubo models estimate --model PATH|CATALOGUE_ID [--json|--jsonl] [--context N] [--sessions N]
```

Both are additive to schema_version 1: two new event types (`models`,
`model_estimate`) and optional fields. Old fixtures stay valid. The commands
run in the engine library, because the engine's own loaders decide which files
it can run. If the runtime is not installed, a machine call returns the typed
`runtime_missing` error (exit 11). `bubo prepare model --plan` needs no
runtime. Exit codes: 0 success, 2 bad arguments, 12 `model_missing` for an
unknown `--model`, 1 unexpected failure.

### Scan rules

`list` scans the model folder (`--models-root`, default
`%LOCALAPPDATA%\Bubo\models`, only if it exists) and each `--scan-folder`. The
user chooses the folders; nothing else is read.

- Depth limit 3: files in the folder, and in folders up to three levels below
  it. A deeper level is skipped and a warning says so. `scan_depth_limit` in
  `listing` repeats the limit.
- A drive root (`C:\`, `C:`), a network share root (`\\server\share`), an
  empty name, a missing folder and a folder that is a link or junction are
  refused. Each refusal is a row in `scan_warnings`.
- Links and junctions inside a folder are never followed, and a folder that was
  already visited is not visited again, so a loop ends. Only a symbolic link
  and a junction count as links. A folder or file that Windows marks in
  another way, for example a OneDrive folder, is scanned like any other.
- A scan stops after 2000 files or 20000 folders, with a warning.
- Only `.gguf` files count. `.partial` downloads are ignored. Of a split model
  (`name-00001-of-00004.gguf`), only shard 1 is listed. Its `size_bytes` is the
  sum of the shards found, and `shards_found` and `shards_expected` show
  missing shards.
- Files that lie inside a catalogue entry's own folder are not repeated under
  `scanned`; `package_files_skipped` counts them.
- A folder that holds `bubo-package.json` is one prepared package, in the model
  folder or in any `--scan-folder`. The scan records it and does not enter it,
  so its files are never `scanned` rows. A package folder is listed under
  `prepared_models` (see below). If its marker cannot be read, or names no
  `package_id`, a row in `scan_warnings` says so and the folder gets no row.

Each `scanned` row has `path`, `file_name`, `folder`, `size_bytes`,
`shards_found`, `shards_expected`, `architecture`, `model_name` (from
`general.name`), `native_context_tokens`, `kind`, `runnable`, `reason`, `route`
and `estimate`.

For each file the CLI reads the header (`general.architecture`, name, context
length) and asks the engine's loaders. `runnable` is true when the engine can
load the file. Otherwise `reason` says why, in plain words, for example that
the engine does not support the architecture. `kind` is `model`, `sidecar`
(a vision projector, or the draft file of a Flash Next or Gemma 4 model) or
`unreadable` (not a GGUF header). `route` names the engine route. The CLI
keeps no list of architectures. A `gemma4` file is runnable (route `gemma4`)
when the Gemma 4 loader's config check (`gemma4::Config::FromGguf`) accepts
its header; serve checks the tensors when it loads the file. A
`gemma4-assistant` file is a sidecar. A `diffusion_gemma` GGUF is not
runnable: DiffusionGemma runs from its safetensors folder.

### Event `models`

`payload.listing` holds `models_root`, `available_bytes` (read like serve
reads it, or null), `scan_depth_limit`, `estimate_defaults`,
`catalogue_version`, `package_files_skipped` and `scanned_folders`. The other
members are `prepared_models` (one row for each catalogue entry that is
`prepared` under `models_root`, then one row for each other package folder the
scan found), `catalogue`, `scanned` and `scan_warnings` (rows with `folder` and
`reason`).

A `prepared_models` row has `id`, `display_name`, `manifest_path`,
`model_path`, `size_bytes`, `prepared_state`, `capabilities` and (in `models
list` only) `catalogue`. `catalogue` is true when the shipped catalogue knows
the marker's `package_id`: the identity, `prepared_state` and `capabilities`
then come from the entry, and the readiness size check runs at that package
folder, whatever its name. It is false for any other marker: the row comes from
the marker (`id` and `display_name` are the `package_id`; `size_bytes` and
`prepared_state` come from the files the marker lists; `context_max` is null).
`prepared_state` is `prepared`, `partial` (some file is missing or has the
wrong size) or `not_prepared`. A package folder outside `models_root` is listed
even if it is partial. The same id in two folders gives two rows; tell them
apart by `manifest_path`. A catalogue folder under `models_root` is listed once,
as before.

In `models list`, a `prepared_models` row also has `repair_output`: the exact
`--output` folder that `bubo prepare --catalogue-id <id>` accepts for that
package, which is the folder above `models/<id>` (prepare's own layout rule).
It is null when the package folder is not in that layout, for example an older
download under another name or an imported folder, and always null when
`catalogue` is false. A client shows Repair only when it is not null.

`capabilities.mtp` is true only for an entry that has a file with the `mtp`
role. `serve` binds that file as `--mtp-model` and decodes with MTP by default
for Flash Next and for Gemma 4 (profiles `default` and `subagents`; `--no-mtp`
turns it off). Without the file, or with the file missing or damaged, the
package serves with MTP off and the startup log says why. Cyber-Tiel has no
`mtp` file and makes no MTP claim.
`manifest_path` is the folder's `bubo-package.json` when it exists, for Flash
Next and for plain catalogue models, else null. `model_path` (nullable string,
added for the GUI) is the absolute path of the main model file (shard 1 for
Flash Next). Start the model with `serve --manifest manifest_path` or
`serve --model model_path`. Status and live serve output use the same row for
the selected or loaded package: its own catalogue id and name (a Flash Next set
or a plain catalogue model); `model_path` is the served file in live serve
output and null in `status`. An id that the catalogue does not know reports the
Huihui package, as before.

Every `catalogue` and `scanned` row has an `estimate` object, described under
`model_estimate`. The list uses 1 session and `min(native context, 32768)`
tokens as its estimate defaults. A row whose estimate fails (for example a
damaged `bubo-package.json`) gets `estimate_state` `none_yet` with the reason
`The estimate failed: ...`; the rest of the list is unchanged.

```jsonl
{"schema_version":1,"type":"models","operation_id":"11111111111111111111111111111111","seq":1,"time":"2026-10-14T12:00:00Z","product":{"name":"bubo","version":"0.1.0","revision":"3ff74dce","engine_version":"0.1.0"},"payload":{"listing":{"models_root":"C:\\Users\\example\\AppData\\Local\\Temp\\bubo-models-test\\list-models","available_bytes":68719476736,"scan_depth_limit":3,"estimate_defaults":{"sessions":1,"context_tokens":"serve release-profile context (native when unset)"},"catalogue_version":"2026-10-14","package_files_skipped":1,"scanned_folders":["C:\\Users\\example\\AppData\\Local\\Temp\\bubo-models-test\\list-models","C:\\Users\\example\\AppData\\Local\\Temp\\bubo-models-test\\my-models","C:\\Users\\example\\AppData\\Local\\Temp\\bubo-models-test\\imported-models"]},"prepared_models":[{"id":"tiny-dense","display_name":"Tiny dense","manifest_path":null,"model_path":"C:\\Users\\example\\AppData\\Local\\Temp\\bubo-models-test\\list-models\\tiny-dense\\tiny.gguf","size_bytes":3645952,"prepared_state":"prepared","capabilities":{"context_max":null,"mtp":false,"twin":false,"mmproj":false},"catalogue":true,"repair_output":null},{"id":"custom-pkg","display_name":"custom-pkg","manifest_path":"C:\\Users\\example\\AppData\\Local\\Temp\\bubo-models-test\\imported-models\\custom-pkg\\bubo-package.json","model_path":"C:\\Users\\example\\AppData\\Local\\Temp\\bubo-models-test\\imported-models\\custom-pkg\\custom.gguf","size_bytes":3646144,"prepared_state":"prepared","capabilities":{"context_max":null,"mtp":false,"twin":false,"mmproj":true},"catalogue":false,"repair_output":null}],"catalogue":[{"id":"tiny-dense","display_name":"Tiny dense","role":"supported","architecture":"qwen35","set":null,"default_status":"none","family_group":null,"download_enabled":true,"disabled_reason":null,"packaging":"gguf","licence":{"id":"MIT","name":"MIT","url":"https://opensource.org/license/mit","prompt_line":"MIT.","note":null,"verified_upstream":true},"files":[{"name":"tiny.gguf","role":"main","bytes":3645952,"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","url":"https://huggingface.co/fixture/tiny/resolve/1111111111111111111111111111111111111111/tiny.gguf","optional":false,"downloadable":true,"licence_text":null}],"download_bytes":3645952,"default_note":null,"estimate_note":null,"speed":null,"stock_comparison":null,"estimate_pointers":[],"prepared_state":"prepared","local_folder":"C:\\Users\\example\\AppData\\Local\\Temp\\bubo-models-test\\list-models\\tiny-dense","estimate":{"estimate_state":"estimated","reason":null,"evidence_label":"estimated from the file header","fit_verified":false,"fit_inputs":{"weights_bytes":3645952,"native_context_tokens":262144,"context_tokens":262144,"sessions":1,"available_bytes":68719476736},"notes":["Attention cache storage uses the shipped production policy (half precision). Not checked on this file.","RAM cache: load-dependent. In this build only the Flash Next runner sizes its cache from the totals. This runner sizes it from free memory when the server starts, so the cache is not in this estimate."],"memory_estimate":{"table_version":1,"evidence":"estimated_from_header","byte_unit":"bytes","display_unit":"GiB","fit_verified":false,"reason":"Estimated from the file header. Scratch memory while the model reads a long prompt and the RAM cache are not included, so the real need is higher. Fit is not verified.","cache_mode":"automatic","fit_scope":"startup_lower_bound","weights":{"bytes":3645952,"gib":0.003395557403564453,"evidence":"estimated"},"per_session_state":{"bytes":2148532224,"gib":2.0009765625,"evidence":"estimated"},"per_session_vision":{"bytes":0,"gib":0,"evidence":"estimated"},"vision_session_total":{"bytes":0,"gib":0,"evidence":"estimated"},"session_state_total":{"bytes":2148532224,"gib":2.0009765625,"evidence":"estimated"},"sessions":1,"context_tokens":262144,"scratch_phase_allowance":{"min":{"bytes":0,"gib":0,"evidence":"unknown"},"max":{"bytes":0,"gib":0,"evidence":"unknown"}},"host_cache_snapshot_allowance":{"bytes":0,"gib":0,"evidence":"estimated"},"reserve":{"bytes":4294967296,"gib":4,"evidence":"estimated"},"total_range":{"min":{"bytes":2152178176,"gib":2.0043721199035645,"evidence":"estimated"},"max":{"bytes":2152178176,"gib":2.0043721199035645,"evidence":"estimated"}},"carve_budget":null,"host_budget":null,"reference_coefficient":{"bytes":2147901440,"gib":2.0003890991210938,"evidence":"estimated","context_tokens":262144,"conditions":"State size of one session at the model's own context length, from the header keys and the engine's state-size functions.","source_path":"src/core/session_state_bytes.hpp"},"device_cache_snapshot_allowance":{"bytes":0,"gib":0,"evidence":"estimated"},"gpu":{"available":{"bytes":68719476736,"gib":64,"evidence":"measured"},"total":null,"dedicated_video_memory":null,"dedicated_video_memory_reason":"DXGI DedicatedVideoMemory is not reported.","dedicated_video_memory_source":"DXGI IDXGIAdapter::GetDesc1 DedicatedVideoMemory","need":{"bytes":2152178176,"gib":2.0043721199035645,"evidence":"estimated"},"need_max":{"bytes":2152178176,"gib":2.0043721199035645,"evidence":"estimated"},"reserve":{"bytes":4294967296,"gib":4,"evidence":"estimated"}},"host":{"available":{"bytes":137438953472,"gib":128,"evidence":"measured"},"total":null,"need":{"bytes":912680550,"gib":0.849999999627471,"evidence":"estimated"},"floor":{"bytes":4294967296,"gib":4,"evidence":"estimated"}},"fits":true,"fit_reason":null,"recommended":true,"recommendation_reason":"Cache can grow to 0.000 GiB; 61.996 GiB left >= margin 5.000 GiB GPU headroom. Host spill is not checked: DXGI DedicatedVideoMemory is not reported. Recommendation uses the cache-growth margin only.","recommendation_failures":[],"cache_at_clamp_bytes":0,"host_spill_bytes":null,"free_after_growth_gib":61.995627880096436,"host_free_after_spill_gib":null,"formula":"weights + session_state_total (lower bound; scratch and RAM cache not included)","recommended_context":{"available_bytes":68719476736,"available_gib":64,"available_evidence":"measured","margin_bytes":5368709120,"margin_gib":5,"margin_rule":"max(5 GiB, 5% of GPU available) after cache growth at the engine clamp; host free after spill must retain its floor. Snapshot reserve is not admission memory.","range_basis":"gpu.need","step_tokens":8192,"minimum_context_tokens":4096,"context_max_tokens":262144,"max_sessions":8,"by_sessions":[{"sessions":1,"context_per_session":262144,"fits":true,"recommended":true,"reason":"Cache can grow to 0.000 GiB; 61.996 GiB left >= margin 5.000 GiB GPU headroom. Host spill is not checked: DXGI DedicatedVideoMemory is not reported. Recommendation uses the cache-growth margin only.","recommendation_failures":[],"cache_at_clamp_bytes":0,"host_spill_bytes":null,"free_after_growth_gib":61.995627880096436,"host_free_after_spill_gib":null,"total_min_bytes":2152178176,"free_after_gib":61.995627880096436},{"sessions":2,"context_per_session":262144,"fits":true,"recommended":true,"reason":"Cache can grow to 0.000 GiB; 59.995 GiB left >= margin 5.000 GiB GPU headroom. Host spill is not checked: DXGI DedicatedVideoMemory is not reported. Recommendation uses the cache-growth margin only.","recommendation_failures":[],"cache_at_clamp_bytes":0,"host_spill_bytes":null,"free_after_growth_gib":59.994651317596436,"host_free_after_spill_gib":null,"total_min_bytes":4300710400,"free_after_gib":59.994651317596436},{"sessions":3,"context_per_session":262144,"fits":true,"recommended":true,"reason":"Cache can grow to 0.000 GiB; 57.994 GiB left >= margin 5.000 GiB GPU headroom. Host spill is not checked: DXGI DedicatedVideoMemory is not reported. Recommendation uses the cache-growth margin only.","recommendation_failures":[],"cache_at_clamp_bytes":0,"host_spill_bytes":null,"free_after_growth_gib":57.993674755096436,"host_free_after_spill_gib":null,"total_min_bytes":6449242624,"free_after_gib":57.993674755096436},{"sessions":4,"context_per_session":262144,"fits":true,"recommended":true,"reason":"Cache can grow to 0.000 GiB; 55.993 GiB left >= margin 5.000 GiB GPU headroom. Host spill is not checked: DXGI DedicatedVideoMemory is not reported. Recommendation uses the cache-growth margin only.","recommendation_failures":[],"cache_at_clamp_bytes":0,"host_spill_bytes":null,"free_after_growth_gib":55.992698192596436,"host_free_after_spill_gib":null,"total_min_bytes":8597774848,"free_after_gib":55.992698192596436},{"sessions":5,"context_per_session":262144,"fits":true,"recommended":true,"reason":"Cache can grow to 0.000 GiB; 53.992 GiB left >= margin 5.000 GiB GPU headroom. Host spill is not checked: DXGI DedicatedVideoMemory is not reported. Recommendation uses the cache-growth margin only.","recommendation_failures":[],"cache_at_clamp_bytes":0,"host_spill_bytes":null,"free_after_growth_gib":53.991721630096436,"host_free_after_spill_gib":null,"total_min_bytes":10746307072,"free_after_gib":53.991721630096436},{"sessions":6,"context_per_session":262144,"fits":true,"recommended":true,"reason":"Cache can grow to 0.000 GiB; 51.991 GiB left >= margin 5.000 GiB GPU headroom. Host spill is not checked: DXGI DedicatedVideoMemory is not reported. Recommendation uses the cache-growth margin only.","recommendation_failures":[],"cache_at_clamp_bytes":0,"host_spill_bytes":null,"free_after_growth_gib":51.990745067596436,"host_free_after_spill_gib":null,"total_min_bytes":12894839296,"free_after_gib":51.990745067596436},{"sessions":7,"context_per_session":262144,"fits":true,"recommended":true,"reason":"Cache can grow to 0.000 GiB; 49.990 GiB left >= margin 5.000 GiB GPU headroom. Host spill is not checked: DXGI DedicatedVideoMemory is not reported. Recommendation uses the cache-growth margin only.","recommendation_failures":[],"cache_at_clamp_bytes":0,"host_spill_bytes":null,"free_after_growth_gib":49.989768505096436,"host_free_after_spill_gib":null,"total_min_bytes":15043371520,"free_after_gib":49.989768505096436},{"sessions":8,"context_per_session":262144,"fits":true,"recommended":true,"reason":"Cache can grow to 0.000 GiB; 47.989 GiB left >= margin 5.000 GiB GPU headroom. Host spill is not checked: DXGI DedicatedVideoMemory is not reported. Recommendation uses the cache-growth margin only.","recommendation_failures":[],"cache_at_clamp_bytes":0,"host_spill_bytes":null,"free_after_growth_gib":47.988791942596436,"host_free_after_spill_gib":null,"total_min_bytes":17191903744,"free_after_gib":47.988791942596436}]},"recommended_context_reason":null}}},{"id":"absent","display_name":"Absent","role":"supported","architecture":"qwen35","set":null,"default_status":"none","family_group":null,"download_enabled":false,"disabled_reason":"not pinned yet","packaging":"gguf","licence":{"id":"MIT","name":"MIT","url":"https://opensource.org/license/mit","prompt_line":"MIT.","note":null,"verified_upstream":true},"files":[{"name":"absent.gguf","role":"main","bytes":0,"sha256":"","url":"","optional":false,"downloadable":true,"licence_text":null}],"download_bytes":0,"default_note":null,"estimate_note":null,"speed":null,"stock_comparison":null,"estimate_pointers":[],"prepared_state":"not_prepared","local_folder":"C:\\Users\\example\\AppData\\Local\\Temp\\bubo-models-test\\list-models\\absent","estimate":{"estimate_state":"none_yet","reason":"The model file is not on this computer, so its header cannot be read.","evidence_label":null,"fit_verified":false,"fit_inputs":{},"notes":[],"memory_estimate":null}}],"scanned":[{"path":"C:\\Users\\example\\AppData\\Local\\Temp\\bubo-models-test\\my-models\\mine.gguf","file_name":"mine.gguf","folder":"C:\\Users\\example\\AppData\\Local\\Temp\\bubo-models-test\\my-models","size_bytes":3645952,"shards_found":1,"shards_expected":1,"architecture":"qwen35","model_name":"Synthetic dense","native_context_tokens":262144,"kind":"model","runnable":true,"reason":null,"route":"qwen","estimate":{"estimate_state":"estimated","reason":null,"evidence_label":"estimated from the file header","fit_verified":false,"fit_inputs":{"weights_bytes":3645952,"native_context_tokens":262144,"context_tokens":262144,"sessions":1,"available_bytes":68719476736},"notes":["Attention cache storage uses the shipped production policy (half precision). Not checked on this file.","RAM cache: load-dependent. In this build only the Flash Next runner sizes its cache from the totals. This runner sizes it from free memory when the server starts, so the cache is not in this estimate."],"memory_estimate":{"table_version":1,"evidence":"estimated_from_header","byte_unit":"bytes","display_unit":"GiB","fit_verified":false,"reason":"Estimated from the file header. Scratch memory while the model reads a long prompt and the RAM cache are not included, so the real need is higher. Fit is not verified.","cache_mode":"automatic","fit_scope":"startup_lower_bound","weights":{"bytes":3645952,"gib":0.003395557403564453,"evidence":"estimated"},"per_session_state":{"bytes":2148532224,"gib":2.0009765625,"evidence":"estimated"},"per_session_vision":{"bytes":0,"gib":0,"evidence":"estimated"},"vision_session_total":{"bytes":0,"gib":0,"evidence":"estimated"},"session_state_total":{"bytes":2148532224,"gib":2.0009765625,"evidence":"estimated"},"sessions":1,"context_tokens":262144,"scratch_phase_allowance":{"min":{"bytes":0,"gib":0,"evidence":"unknown"},"max":{"bytes":0,"gib":0,"evidence":"unknown"}},"host_cache_snapshot_allowance":{"bytes":0,"gib":0,"evidence":"estimated"},"reserve":{"bytes":4294967296,"gib":4,"evidence":"estimated"},"total_range":{"min":{"bytes":2152178176,"gib":2.0043721199035645,"evidence":"estimated"},"max":{"bytes":2152178176,"gib":2.0043721199035645,"evidence":"estimated"}},"carve_budget":null,"host_budget":null,"reference_coefficient":{"bytes":2147901440,"gib":2.0003890991210938,"evidence":"estimated","context_tokens":262144,"conditions":"State size of one session at the model's own context length, from the header keys and the engine's state-size functions.","source_path":"src/core/session_state_bytes.hpp"},"device_cache_snapshot_allowance":{"bytes":0,"gib":0,"evidence":"estimated"},"gpu":{"available":{"bytes":68719476736,"gib":64,"evidence":"measured"},"total":null,"dedicated_video_memory":null,"dedicated_video_memory_reason":"DXGI DedicatedVideoMemory is not reported.","dedicated_video_memory_source":"DXGI IDXGIAdapter::GetDesc1 DedicatedVideoMemory","need":{"bytes":2152178176,"gib":2.0043721199035645,"evidence":"estimated"},"need_max":{"bytes":2152178176,"gib":2.0043721199035645,"evidence":"estimated"},"reserve":{"bytes":4294967296,"gib":4,"evidence":"estimated"}},"host":{"available":{"bytes":137438953472,"gib":128,"evidence":"measured"},"total":null,"need":{"bytes":912680550,"gib":0.849999999627471,"evidence":"estimated"},"floor":{"bytes":4294967296,"gib":4,"evidence":"estimated"}},"fits":true,"fit_reason":null,"recommended":true,"recommendation_reason":"Cache can grow to 0.000 GiB; 61.996 GiB left >= margin 5.000 GiB GPU headroom. Host spill is not checked: DXGI DedicatedVideoMemory is not reported. Recommendation uses the cache-growth margin only.","recommendation_failures":[],"cache_at_clamp_bytes":0,"host_spill_bytes":null,"free_after_growth_gib":61.995627880096436,"host_free_after_spill_gib":null,"formula":"weights + session_state_total (lower bound; scratch and RAM cache not included)","recommended_context":{"available_bytes":68719476736,"available_gib":64,"available_evidence":"measured","margin_bytes":5368709120,"margin_gib":5,"margin_rule":"max(5 GiB, 5% of GPU available) after cache growth at the engine clamp; host free after spill must retain its floor. Snapshot reserve is not admission memory.","range_basis":"gpu.need","step_tokens":8192,"minimum_context_tokens":4096,"context_max_tokens":262144,"max_sessions":8,"by_sessions":[{"sessions":1,"context_per_session":262144,"fits":true,"recommended":true,"reason":"Cache can grow to 0.000 GiB; 61.996 GiB left >= margin 5.000 GiB GPU headroom. Host spill is not checked: DXGI DedicatedVideoMemory is not reported. Recommendation uses the cache-growth margin only.","recommendation_failures":[],"cache_at_clamp_bytes":0,"host_spill_bytes":null,"free_after_growth_gib":61.995627880096436,"host_free_after_spill_gib":null,"total_min_bytes":2152178176,"free_after_gib":61.995627880096436},{"sessions":2,"context_per_session":262144,"fits":true,"recommended":true,"reason":"Cache can grow to 0.000 GiB; 59.995 GiB left >= margin 5.000 GiB GPU headroom. Host spill is not checked: DXGI DedicatedVideoMemory is not reported. Recommendation uses the cache-growth margin only.","recommendation_failures":[],"cache_at_clamp_bytes":0,"host_spill_bytes":null,"free_after_growth_gib":59.994651317596436,"host_free_after_spill_gib":null,"total_min_bytes":4300710400,"free_after_gib":59.994651317596436},{"sessions":3,"context_per_session":262144,"fits":true,"recommended":true,"reason":"Cache can grow to 0.000 GiB; 57.994 GiB left >= margin 5.000 GiB GPU headroom. Host spill is not checked: DXGI DedicatedVideoMemory is not reported. Recommendation uses the cache-growth margin only.","recommendation_failures":[],"cache_at_clamp_bytes":0,"host_spill_bytes":null,"free_after_growth_gib":57.993674755096436,"host_free_after_spill_gib":null,"total_min_bytes":6449242624,"free_after_gib":57.993674755096436},{"sessions":4,"context_per_session":262144,"fits":true,"recommended":true,"reason":"Cache can grow to 0.000 GiB; 55.993 GiB left >= margin 5.000 GiB GPU headroom. Host spill is not checked: DXGI DedicatedVideoMemory is not reported. Recommendation uses the cache-growth margin only.","recommendation_failures":[],"cache_at_clamp_bytes":0,"host_spill_bytes":null,"free_after_growth_gib":55.992698192596436,"host_free_after_spill_gib":null,"total_min_bytes":8597774848,"free_after_gib":55.992698192596436},{"sessions":5,"context_per_session":262144,"fits":true,"recommended":true,"reason":"Cache can grow to 0.000 GiB; 53.992 GiB left >= margin 5.000 GiB GPU headroom. Host spill is not checked: DXGI DedicatedVideoMemory is not reported. Recommendation uses the cache-growth margin only.","recommendation_failures":[],"cache_at_clamp_bytes":0,"host_spill_bytes":null,"free_after_growth_gib":53.991721630096436,"host_free_after_spill_gib":null,"total_min_bytes":10746307072,"free_after_gib":53.991721630096436},{"sessions":6,"context_per_session":262144,"fits":true,"recommended":true,"reason":"Cache can grow to 0.000 GiB; 51.991 GiB left >= margin 5.000 GiB GPU headroom. Host spill is not checked: DXGI DedicatedVideoMemory is not reported. Recommendation uses the cache-growth margin only.","recommendation_failures":[],"cache_at_clamp_bytes":0,"host_spill_bytes":null,"free_after_growth_gib":51.990745067596436,"host_free_after_spill_gib":null,"total_min_bytes":12894839296,"free_after_gib":51.990745067596436},{"sessions":7,"context_per_session":262144,"fits":true,"recommended":true,"reason":"Cache can grow to 0.000 GiB; 49.990 GiB left >= margin 5.000 GiB GPU headroom. Host spill is not checked: DXGI DedicatedVideoMemory is not reported. Recommendation uses the cache-growth margin only.","recommendation_failures":[],"cache_at_clamp_bytes":0,"host_spill_bytes":null,"free_after_growth_gib":49.989768505096436,"host_free_after_spill_gib":null,"total_min_bytes":15043371520,"free_after_gib":49.989768505096436},{"sessions":8,"context_per_session":262144,"fits":true,"recommended":true,"reason":"Cache can grow to 0.000 GiB; 47.989 GiB left >= margin 5.000 GiB GPU headroom. Host spill is not checked: DXGI DedicatedVideoMemory is not reported. Recommendation uses the cache-growth margin only.","recommendation_failures":[],"cache_at_clamp_bytes":0,"host_spill_bytes":null,"free_after_growth_gib":47.988791942596436,"host_free_after_spill_gib":null,"total_min_bytes":17191903744,"free_after_gib":47.988791942596436}]},"recommended_context_reason":null}}},{"path":"C:\\Users\\example\\AppData\\Local\\Temp\\bubo-models-test\\my-models\\other.gguf","file_name":"other.gguf","folder":"C:\\Users\\example\\AppData\\Local\\Temp\\bubo-models-test\\my-models","size_bytes":192,"shards_found":1,"shards_expected":1,"architecture":"gemma4","model_name":"Synthetic gemma4","native_context_tokens":131072,"kind":"model","runnable":false,"reason":"The engine cannot load this gemma4 file: missing or invalid GGUF integer gemma4.embedding_length","route":null,"estimate":{"estimate_state":"none_yet","reason":"The engine cannot load this gemma4 file: missing or invalid GGUF integer gemma4.embedding_length","evidence_label":null,"fit_verified":false,"fit_inputs":{},"notes":[],"memory_estimate":null}},{"path":"C:\\Users\\example\\AppData\\Local\\Temp\\bubo-models-test\\my-models\\proj.gguf","file_name":"proj.gguf","folder":"C:\\Users\\example\\AppData\\Local\\Temp\\bubo-models-test\\my-models","size_bytes":192,"shards_found":1,"shards_expected":1,"architecture":"clip","model_name":"Synthetic clip","native_context_tokens":131072,"kind":"sidecar","runnable":false,"reason":"This file is a vision projector. Use it as a side file of a model.","route":null,"estimate":{"estimate_state":"none_yet","reason":"This file is a vision projector. Use it as a side file of a model.","evidence_label":null,"fit_verified":false,"fit_inputs":{},"notes":[],"memory_estimate":null}}],"scan_warnings":[{"folder":"C:\\","reason":"A drive or share root is never scanned. Choose a model folder."}]}}
```

### The catalogue

The catalogue is a file compiled into the program (`src/cli/model_catalogue.json`
and the three Flash Next sets from the package pins). There is no runtime
override. Each `catalogue` row has:

| Field | Meaning |
|---|---|
| `id` | Catalogue id. Use it with `prepare --catalogue-id` and `estimate --model`. For Flash Next it is the package id. |
| `display_name`, `architecture` | Name for lists, and `general.architecture` (or the `model_type` of a folder). |
| `technical_name`, `about`, `about_links`, `selection_label` | CLI-owned model identity, plain card text, HTTPS source links and optional short label for the first-run select. Cards show no speed figures. The lower-memory card carries only the explicitly attributed quality statement about ISTA's source file. |
| `gpu_bytes`, `system_memory_bytes` | Weight-placement bytes, both null unless exact tensor arithmetic or a cited measurement is available. They exclude conversation caches and work buffers. Missing placement means show download size only. |
| `system_memory_note`, `size_split_source` | Name of the part kept in system memory and the source of the split; null without a measured split. |
| `role` | `recommended` or `supported` (including the lowmemory option). |
| `set` | `uncensored`, `original` or null. |
| `default_status` | `default`, `to be decided` or `none`. |
| `family_group` | Entries that are alternatives of one model share the value, for example the two Gemma 4 26B-A4B files. |
| `download_enabled`, `disabled_reason` | An entry is downloadable only with a full pin. A disabled entry has a reason. |
| `packaging` | `gguf`, or `safetensors_folder` (DiffusionGemma). |
| `licence` | `id`, `name`, `url` (https), `prompt_line` (the exact line the download prompt shows), `note` (extra caution, or null) and `verified_upstream` (true when the upstream licence file was read). |
| `files` | Rows with `name`, `role` (main, mtp, draft_head, twin, mmproj, support), `bytes`, `sha256`, `url`, `optional`, `downloadable` and `licence_text`. A file with `downloadable` false is never fetched. Each Gemma QAT entry includes its model, vision projector and MTP head from one pinned maker repository. A downloadable file from another repository carries its own licence line. |
| `download_bytes` | Sum of the downloadable files. |
| `speed` | Measured speed, or null. `writing_tok_s` (decode), `reading_tok_s` (prefill), `prompt_tokens`, `source` and `label`. A number that was not measured is null. The whole object is null when nothing was measured. Never guessed. |
| `stock_comparison` | The same model on stock llama.cpp, or null: `engine`, `prefill_pp8192`, `decode_tg128` and `source` (a llama-bench measurement). Data only; the client decides whether to show it. |
| `prepared_state` | `prepared` (every downloadable file is present with the exact size), `partial` or `not_prepared`. Size check only; serve keeps the full checks. The projector counts: a model without it is `partial`, and the row's estimate notes name `mmproj-BF16.gguf`. |
| `local_folder` | `models_root\<id>`. |
| `estimate_note`, `estimate_pointers` | Why there is no estimate yet, and the engine source lines to use after a later merge. |
| `default_note` | A plain catalogue note, including default rationale or local-only availability, or null. |

Pin rules, enforced when the catalogue loads (and tested): an entry can have
`download_enabled` true only if every downloadable file has an https URL equal
to `https://huggingface.co/REPO/resolve/COMMIT/PATH` for a repository and a
40-digit commit, a byte count and a SHA-256, and the entry has a licence `id`,
`name`, https `url` and `prompt_line`. If `verified_upstream` is false, the
licence needs a `note`. A bad catalogue stops the program with an error; it
does not fall back.

The shipped catalogue has eleven entries, including four Gemma QAT builds from
Unsloth and Huihui and the NVFP4 DiffusionGemma build. DiffusionGemma includes the
pinned HaloForge Labs vision conversion and conversion notice; its download is enabled.
`huihui-qwen38-flash-next-lowmemory-v1` is named
"Qwen3.8 Flash Next · HaloForge Labs lower-memory quant (GSQ-RCO)", with `role` `supported`
and `default_status` `none`. The default remains
`huihui-qwen38-flash-next-ud-q4-k-xl-v1`. The lowmemory replaces four main shards;
MTP, mmproj, twin and draft head retain the Huihui pins. Speed and stock
comparison are null. Its independent measured resident coefficient is unset,
so its estimate is `none_yet` with no memory verdict.

The lower-memory package is public and enabled in the release catalogue;
there is no publication constant or gate. All three Flash Next options are
enabled, including Get started. The host repository is
`HaloForgeLabs/bubo-qwen3.8-flash-next-lower-memory`.
Tests use local files, recorded plans or an
in-process fixture transport; they never download from that host. A complete
local set can still be reused. `lowmemory_data.hpp` pins its full revision.
Catalogue destinations use the uploaded
`Qwen3.8-Flash-Next-Huihui-LowMemory-0000N-of-00004.gguf` shard names. If a required
file is absent or damaged in offline mode, prepare refuses acquisition with
exit code 4 before any network use. A complete local set at
`ROOT/models/huihui-qwen38-flash-next-lowmemory-v1` can be checked with
`bubo prepare model --catalogue-id huihui-qwen38-flash-next-lowmemory-v1 --output ROOT --plan --offline`.
The lowmemory has separate generated local-composition evidence; it does not
claim published HF provenance. NOTICE carries the release attribution
for Qwen, Huihui, Unsloth, ISTA-DASLab and HaloForge Labs.

### Event `model_estimate`

`estimate --model` takes a GGUF path, a folder, or a catalogue id. A folder
that holds a `bubo-package.json` of a shipped catalogue entry gets the estimate
of that entry, as if you gave the catalogue id: a Flash Next folder keeps its
measured table (`fit_scope` `conservative_startup_estimate`), and a GGUF
package gets the header estimate. This holds for a package in any folder, with
any folder name. The payload
has `subject` and `subject_kind` (`path` or `catalogue`) and the estimate
fields below. `--profile` selects the same release profile as serve (default when
absent); unset context uses that profile or the native header context. The same
resolved settings build both admission and recommendation inputs.
`estimate_state` is `estimated` or `none_yet`. A GUI shows
"no estimate yet" and the `reason` for `none_yet`.

- `memory_estimate` is the existing `memory_estimate` object, including
  `recommended_context`. It is null when `estimate_state` is `none_yet`.
- `reason` is null for `estimated`.
- `evidence_label` says where the numbers come from, for example
  `estimated from the file header`.
- `fit_verified` is always false. `fit_inputs` lists what went in:
  `weights_bytes`, `native_context_tokens`, `context_tokens`, `sessions`,
  `available_bytes`.
- `notes` lists caveats, for example that the RAM cache is not in the total.

How the numbers are made:

- Flash Next (the catalogue ids, or a `qwen4exp` file that the engine's loader
  accepts): the measured coefficient table, the same estimator as
  `serve --print-effective-settings`. Prepared packages use the same GPU-resident
  weight coefficients regardless of selection by id, folder or main-shard path.
  Unpackaged files use their own main-file sizes plus selected side coefficients.
- Dense Qwen (`qwen35`), from the header keys: the engine's own
  `QwenDenseSessionStateBytes` (KV cache for the attention layers, convolution
  and recurrent state for the others), with the production half-precision
  cache. The same function sizes the GPU allocation.
- Qwen3.6-35B-A3B (`qwen35moe`): the engine's `A3bSessionStateBytes` and its
  `ReadConfig`. This runtime serves one session.
- Header estimates carry `evidence` `estimated_from_header`. The total is a
  lower bound: weights (the sum of the file sizes of the selected files), the
  session state. The 4 GiB reserve only sizes the snapshot pool. Scratch memory and the RAM cache are
  not in it (`scratch_phase_allowance` has `evidence` `unknown`). In this build
  only the Flash Next runner sizes its snapshot cache from the totals; the
  other runners size it from free memory at server start.
- A bound vision projector (an explicit `--mmproj`, or the prepared projector
  of a plain catalogue model) on the dense Qwen and Qwen3.6-A3B routes adds to
  `weights` the bytes that `qwen::vision::Encoder` uploads on the first image:
  every projector tensor except the two F32 patch-embedding halves (4 bytes
  per F32 element, 2 otherwise) plus the fused BF16 patch weight (1152 x 1536
  values). A note names the file and the bytes. These runners reserve nothing
  for vision at startup. For a catalogue id, the base `weights` term and
  `fit_inputs.weights_bytes` are the model files only.
- A file whose pattern cannot be read gives `none_yet` and a reason. There is
  no guess of full attention. DeepSeek, Gemma 4 and DiffusionGemma have no
  CPU-side memory estimate in this build, so they give `none_yet`; the
  catalogue keeps the engine source lines for the later merge in
  `estimate_pointers`. The engine runs these models; `none_yet` says only that
  the estimate is missing. The `estimate_note` of the Gemma 4 entries with a
  projector names the 2 GiB that the Gemma 4 runner reserves for a bound
  projector.

```jsonl
{"schema_version":1,"type":"model_estimate","operation_id":"11111111111111111111111111111111","seq":1,"time":"2026-10-14T12:00:00Z","product":{"name":"bubo","version":"0.1.0","revision":"3ff74dce","engine_version":"0.1.0"},"payload":{"estimate_state":"estimated","reason":null,"evidence_label":"estimated from the file header","fit_verified":false,"fit_inputs":{"weights_bytes":3645952,"native_context_tokens":262144,"context_tokens":262144,"sessions":1,"available_bytes":68719476736},"notes":["Attention cache storage uses the shipped production policy (half precision). Not checked on this file.","RAM cache: load-dependent. In this build only the Flash Next runner sizes its cache from the totals. This runner sizes it from free memory when the server starts, so the cache is not in this estimate."],"memory_estimate":{"table_version":1,"evidence":"estimated_from_header","byte_unit":"bytes","display_unit":"GiB","fit_verified":false,"reason":"Estimated from the file header. Scratch memory while the model reads a long prompt and the RAM cache are not included, so the real need is higher. Fit is not verified.","cache_mode":"automatic","fit_scope":"startup_lower_bound","weights":{"bytes":3645952,"gib":0.003395557403564453,"evidence":"estimated"},"per_session_state":{"bytes":2148532224,"gib":2.0009765625,"evidence":"estimated"},"per_session_vision":{"bytes":0,"gib":0,"evidence":"estimated"},"vision_session_total":{"bytes":0,"gib":0,"evidence":"estimated"},"session_state_total":{"bytes":2148532224,"gib":2.0009765625,"evidence":"estimated"},"sessions":1,"context_tokens":262144,"scratch_phase_allowance":{"min":{"bytes":0,"gib":0,"evidence":"unknown"},"max":{"bytes":0,"gib":0,"evidence":"unknown"}},"host_cache_snapshot_allowance":{"bytes":0,"gib":0,"evidence":"estimated"},"reserve":{"bytes":4294967296,"gib":4,"evidence":"estimated"},"total_range":{"min":{"bytes":2152178176,"gib":2.0043721199035645,"evidence":"estimated"},"max":{"bytes":2152178176,"gib":2.0043721199035645,"evidence":"estimated"}},"carve_budget":null,"host_budget":null,"reference_coefficient":{"bytes":2147901440,"gib":2.0003890991210938,"evidence":"estimated","context_tokens":262144,"conditions":"State size of one session at the model's own context length, from the header keys and the engine's state-size functions.","source_path":"src/core/session_state_bytes.hpp"},"device_cache_snapshot_allowance":{"bytes":0,"gib":0,"evidence":"estimated"},"gpu":{"available":{"bytes":68719476736,"gib":64,"evidence":"measured"},"total":null,"dedicated_video_memory":null,"dedicated_video_memory_reason":"DXGI DedicatedVideoMemory is not reported.","dedicated_video_memory_source":"DXGI IDXGIAdapter::GetDesc1 DedicatedVideoMemory","need":{"bytes":2152178176,"gib":2.0043721199035645,"evidence":"estimated"},"need_max":{"bytes":2152178176,"gib":2.0043721199035645,"evidence":"estimated"},"reserve":{"bytes":4294967296,"gib":4,"evidence":"estimated"}},"host":{"available":{"bytes":137438953472,"gib":128,"evidence":"measured"},"total":null,"need":{"bytes":912680550,"gib":0.849999999627471,"evidence":"estimated"},"floor":{"bytes":4294967296,"gib":4,"evidence":"estimated"}},"fits":true,"fit_reason":null,"recommended":true,"recommendation_reason":"Cache can grow to 0.000 GiB; 61.996 GiB left >= margin 5.000 GiB GPU headroom. Host spill is not checked: DXGI DedicatedVideoMemory is not reported. Recommendation uses the cache-growth margin only.","recommendation_failures":[],"cache_at_clamp_bytes":0,"host_spill_bytes":null,"free_after_growth_gib":61.995627880096436,"host_free_after_spill_gib":null,"formula":"weights + session_state_total (lower bound; scratch and RAM cache not included)","recommended_context":{"available_bytes":68719476736,"available_gib":64,"available_evidence":"measured","margin_bytes":5368709120,"margin_gib":5,"margin_rule":"max(5 GiB, 5% of GPU available) after cache growth at the engine clamp; host free after spill must retain its floor. Snapshot reserve is not admission memory.","range_basis":"gpu.need","step_tokens":8192,"minimum_context_tokens":4096,"context_max_tokens":262144,"max_sessions":8,"by_sessions":[{"sessions":1,"context_per_session":262144,"fits":true,"recommended":true,"reason":"Cache can grow to 0.000 GiB; 61.996 GiB left >= margin 5.000 GiB GPU headroom. Host spill is not checked: DXGI DedicatedVideoMemory is not reported. Recommendation uses the cache-growth margin only.","recommendation_failures":[],"cache_at_clamp_bytes":0,"host_spill_bytes":null,"free_after_growth_gib":61.995627880096436,"host_free_after_spill_gib":null,"total_min_bytes":2152178176,"free_after_gib":61.995627880096436},{"sessions":2,"context_per_session":262144,"fits":true,"recommended":true,"reason":"Cache can grow to 0.000 GiB; 59.995 GiB left >= margin 5.000 GiB GPU headroom. Host spill is not checked: DXGI DedicatedVideoMemory is not reported. Recommendation uses the cache-growth margin only.","recommendation_failures":[],"cache_at_clamp_bytes":0,"host_spill_bytes":null,"free_after_growth_gib":59.994651317596436,"host_free_after_spill_gib":null,"total_min_bytes":4300710400,"free_after_gib":59.994651317596436},{"sessions":3,"context_per_session":262144,"fits":true,"recommended":true,"reason":"Cache can grow to 0.000 GiB; 57.994 GiB left >= margin 5.000 GiB GPU headroom. Host spill is not checked: DXGI DedicatedVideoMemory is not reported. Recommendation uses the cache-growth margin only.","recommendation_failures":[],"cache_at_clamp_bytes":0,"host_spill_bytes":null,"free_after_growth_gib":57.993674755096436,"host_free_after_spill_gib":null,"total_min_bytes":6449242624,"free_after_gib":57.993674755096436},{"sessions":4,"context_per_session":262144,"fits":true,"recommended":true,"reason":"Cache can grow to 0.000 GiB; 55.993 GiB left >= margin 5.000 GiB GPU headroom. Host spill is not checked: DXGI DedicatedVideoMemory is not reported. Recommendation uses the cache-growth margin only.","recommendation_failures":[],"cache_at_clamp_bytes":0,"host_spill_bytes":null,"free_after_growth_gib":55.992698192596436,"host_free_after_spill_gib":null,"total_min_bytes":8597774848,"free_after_gib":55.992698192596436},{"sessions":5,"context_per_session":262144,"fits":true,"recommended":true,"reason":"Cache can grow to 0.000 GiB; 53.992 GiB left >= margin 5.000 GiB GPU headroom. Host spill is not checked: DXGI DedicatedVideoMemory is not reported. Recommendation uses the cache-growth margin only.","recommendation_failures":[],"cache_at_clamp_bytes":0,"host_spill_bytes":null,"free_after_growth_gib":53.991721630096436,"host_free_after_spill_gib":null,"total_min_bytes":10746307072,"free_after_gib":53.991721630096436},{"sessions":6,"context_per_session":262144,"fits":true,"recommended":true,"reason":"Cache can grow to 0.000 GiB; 51.991 GiB left >= margin 5.000 GiB GPU headroom. Host spill is not checked: DXGI DedicatedVideoMemory is not reported. Recommendation uses the cache-growth margin only.","recommendation_failures":[],"cache_at_clamp_bytes":0,"host_spill_bytes":null,"free_after_growth_gib":51.990745067596436,"host_free_after_spill_gib":null,"total_min_bytes":12894839296,"free_after_gib":51.990745067596436},{"sessions":7,"context_per_session":262144,"fits":true,"recommended":true,"reason":"Cache can grow to 0.000 GiB; 49.990 GiB left >= margin 5.000 GiB GPU headroom. Host spill is not checked: DXGI DedicatedVideoMemory is not reported. Recommendation uses the cache-growth margin only.","recommendation_failures":[],"cache_at_clamp_bytes":0,"host_spill_bytes":null,"free_after_growth_gib":49.989768505096436,"host_free_after_spill_gib":null,"total_min_bytes":15043371520,"free_after_gib":49.989768505096436},{"sessions":8,"context_per_session":262144,"fits":true,"recommended":true,"reason":"Cache can grow to 0.000 GiB; 47.989 GiB left >= margin 5.000 GiB GPU headroom. Host spill is not checked: DXGI DedicatedVideoMemory is not reported. Recommendation uses the cache-growth margin only.","recommendation_failures":[],"cache_at_clamp_bytes":0,"host_spill_bytes":null,"free_after_growth_gib":47.988791942596436,"host_free_after_spill_gib":null,"total_min_bytes":17191903744,"free_after_gib":47.988791942596436}]},"recommended_context_reason":null},"subject":"C:\\Users\\example\\AppData\\Local\\Temp\\bubo-models-test\\gguf\\dense.gguf","subject_kind":"path"}}
{"schema_version":1,"type":"model_estimate","operation_id":"11111111111111111111111111111111","seq":1,"time":"2026-10-14T12:00:00Z","product":{"name":"bubo","version":"0.1.0","revision":"3ff74dce","engine_version":"0.1.0"},"payload":{"estimate_state":"none_yet","reason":"This build runs Gemma 4, but it has no memory estimate for Gemma 4 yet. The pointers name the engine code for that estimate. A bound BF16 vision projector reserves 2 GiB for its tower and image-sized scratch at serve startup.","evidence_label":null,"fit_verified":false,"fit_inputs":{},"notes":["Source for a later merge: engine tree r7ent ab09274f: src/models/gemma4/config.cpp:144-167,230-245","Source for a later merge: engine tree r7ent ab09274f: src/models/gemma4/kernels/rocm/executor.cpp:184-226 (sliding layers ring = ceil256(window + max_rows); global layers max_context)"],"memory_estimate":null,"subject":"gemma4-26b-a4b-uncensored-hauhaucs-qat-q4-k-m","subject_kind":"catalogue"}}
```

### Prepare by catalogue id and plan

`prepare model --catalogue-id ID` prepares one catalogue entry. The two Flash
Next ids use the existing fixed packages (same files, folders and legal texts
as `--model-set`). Every other id uses the generic path with the same plan,
consent, download, verify and repair code: files go to
`<output>\models\<id>\`, the licence block is written to
`licenses\LICENSE-INFO.txt`, and the package marker `bubo-package.json` names
the id. The saved default model does not change. A disabled entry stops with
exit 4 before any network use. `--catalogue-id` cannot be used with
`--model-set`, and `--mtp-model` and `--mmproj` apply to Flash Next ids only.

`--plan` (model step) shows the plan and stops: files, sizes, the free space of
the target drive and the licence block. It writes nothing and uses no network.
In machine mode it emits the usual `plan` records and a `result` with
`plan_only` true and `prepared` set to whether the files are already ready.
`--plan` does not stop when the drive is too small. The volume rows carry
`free_bytes` and `required_bytes`, so a client can show the shortage. Human mode
prints the plan and then exits 7; machine mode returns exit 0 and the client
compares the two numbers.
The licence block is part of the plan digest, and the licence text is shown
before the first download in every mode. `--plan` asks no question; the plan
ends after its last line. The files under `licenses\` are written by the
download, so the plan names the folder ("A copy is saved in ... after the
download"). In human mode, a catalogue prepare ends with the command that
starts the model: `bubo serve llm --manifest "<output>\models\<id>"`, or
`--model` for the DiffusionGemma folder. A package marker that does not match
this build (Flash Next sets) stops with exit 6 (`integrity_failed`). A model file
in the package folder that differs from its pin is repaired instead. The plan
lists it as a download, shows its size in the consent, and names it in
`replaces`. After approval Bubo downloads the file again beside the old one
(`<name>.partial`) with the same resume and SHA-256 checks as a new file. The
old file keeps its name until the new one has passed its check, so a failed,
cancelled, or killed repair changes nothing and leaves nothing to clean up; run
prepare again. Then Bubo swaps by renames only: it renames the old file to
`<name>.damaged-<UTC stamp>`, renames the new file into place, and deletes the
renamed copy. A rename never writes into the old file, so another hard link to
the same data keeps its bytes. If the old file changed since the plan, prepare
stops with exit 3 (plan changed). If a rename fails because another program
holds the file, the error tells the user to stop the model server and run
prepare again. A file the user supplied (`--mmproj`, `--mtp-model`) that does not match its
pin, and a pinned file whose real location is outside the package folder (for
example through a folder link), still stop with exit 6. With `--offline` a
damaged file stops with exit 4.

A vision projector is a file of its entry: prepare downloads it next to the
model file and the plan lists it like every other file. A downloadable file
with its own licence line (every file from another repository, such as the
Gemma 4 31B projector, and the HauhauCS 26B-A4B projector and MTP head) also gets
`licenses\LICENSE-INFO-<file name>.txt`, with its own licence line, file name,
source repository and revision. The plan shows that block before approval and
the machine plan lists it as a further `licenses` row. A package marker
written for an older catalogue (for example before the entry listed its
projector) does not stop prepare: it fetches the missing files only and writes
the current marker last.

### Serving a plain catalogue model

`bubo serve --model <models_root>\<id>\<model file>` or
`bubo serve --manifest <models_root>\<id>\bubo-package.json` serves a prepared
plain catalogue model. Serve checks its package marker against the compiled
pins: family, source repository and revision, file names, sizes and SHA-256
values. A marker can name fewer side files than the current catalogue. The
`mmproj` file in the same folder becomes the `--mmproj` default when it is
present with its pinned size and a readable GGUF header. This also applies to
a catalogue folder without its marker. The setting row has origin `manifest`,
and `engine.mmproj_bound` of the resolved settings
(`serve --print-effective-settings`) names the file. `--no-vision`,
`--mmproj PATH` and `BUBO_MMPROJ` keep their precedence. Without the projector,
serve starts text only and logs one `vision:` line that names the missing file
and the prepare command. With the marker, `active_model.catalogue_id` is the
entry id. `engine.mmproj_bound` and `engine.vision_disabled` report the
projector for every model family, not only for Flash Next.

## Engine registry

Run `bubo serve --list-switches --json`. This works without a model or engine
load. It returns a switches event whose payload matches switches.golden.json.
Every row has name, kind, class, purpose, group, audience, values per profile,
allowed values/range, hardware_scope, requires, suppressed_by, note,
experimental, restart_required and secret. `secret` defaults to false and marks
values that must never be echoed. It is independent of `echo`, which controls
whether a value may be printed. The generator owns the group rule table
and all generated goldens. Live `effective_switches` has one row per registry
switch with requested, effective, source and forced reason. Null means no
available or safe value. Inference defaults and bound endpoint values replace
unresolved defaults. Hidden environment values are never echoed.

User switches appear in Main. Dense twin and NPU prefill are adjacent in
Hardware and memory. Advanced holds MTP/draft head, lane-row, prompt lookup
and Grouped expert kernel (decode). The latter is experimental and off by
default. Debug switches appear in the collapsed troubleshooting subsection
at the end of Advanced, with purpose, default and warning. See
docs/SWITCH-AUDIENCE.md and engine-panels.json for assignments.
`gui_control` states whether a registry row can be a GUI control. It is false
for the four existing budget/force-close diagnostics under the no-budget/cap/
force-close rule. Other debug rows remain in the collapsed subsection. The
complete CLI listing retains all rows. Clients use this field; they do not
hard-code the excluded names. This applies the stronger control prohibition
to the later instruction to show every debug row.
`docs/switch-evidence.json` maps each name to a cited measured record or null.
Null means not measured. The Oct 8 twin-plus-draft-head record does not isolate
twin-only speed or its GPU-memory cost; do not display the user's
twin-only figures as measurements.

## Local support ZIP

Run `bubo diagnose --bundle ABSOLUTE.zip --json --preview` to see the file
list, exact text and hashes. Preview creates no file. Use the same command
without --preview to save a new ZIP. `diagnose --json --preview` also works.
The route runs before engine load. It never uploads. Only the named new ZIP
path is written; existing targets, links, network/device paths, reserved
Windows names, and traversal are rejected. Parent directories are not created.

The allow-list is versions.json, settings.json, status.json, diagnose.json,
history.json, logs.json and manifest.json. Projection retains typed enums,
bounded numeric settings, known catalogue switch names and compile versions.
Versions contains SHA-256 of the local bootstrap and engine binaries, or null
if a fixed binary is missing or cannot be read. Binary reads are bounded to
512 MiB; the binary and its path are never copied into the bundle.
Paths, user and host identity, raw error messages, serials, fingerprints,
client IPs, tokens, proxy credentials, arbitrary environment values, prompts,
outputs, request bodies, content traces, Pi and auth configuration are omitted.
Unknown raw log lines are dropped whole. No completeness claim is based on
regex. The exported log entries have known event names and bounded counters.
State history holds at most 32 projected transitions. A missing persisted
diagnose result uses a labelled status-readiness summary; it does not pretend
to be a hardware diagnosis. Sanitised effective settings come from verified
CLI status. Saved GUI preferences can supply a separate unavailable-server
fallback. No file contents are copied without typed projection.

Each member is at most 32768 bytes, total text at most 131072 bytes and ZIP at
most 163840 bytes. Source JSON reads are at most 65536 bytes, each log tail at
most 32768 bytes, and exported logs retain at most 256 records. The exact
same builder produces preview and saved content. SHA-256 covers every member
and the whole deterministic ZIP. Changes between commands can change hashes;
clients must display a new preview when content changes. Output records retain
the 1 MiB limit. Save success uses exit 0; invalid arguments use 2; storage
failure uses existing storage_failed/7. The bundle never overwrites a file.

### Event `switches`

Typed complete registry. This compact example shows one row; the full listing fixture contains every row.

```jsonl
{"schema_version":1,"operation_id":"11111111111111111111111111111111","seq":1,"time":"2026-10-13T18:00:00Z","product":{"name":"bubo","version":"0.1.0","revision":"3ff74dce","engine_version":"0.1.0"},"type":"switches","payload":{"schema_version":1,"switches":[{"name":"--allow-unmeasured-memory","kind":"flag","class":"OPTION","model_family":"any","platform":"any","requires":[],"values":{"default":null,"subagents":null,"exact":null},"exact_off":false,"inject_with":null,"suppressed_by":[],"purpose":"Allow subagent context above the measured token limit.","default_text":"Unset","hardware_scope":"generic","note":"Keep it off unless the required context exceeds the measured memory limit.","built_in":null,"value_rule":"text","aliases":[],"takes_value":false,"echo":false,"group":"Memory","audience":"debug","experimental":false,"restart_required":true,"allowed":{"type":"boolean","values":[false,true]},"gui_control":true}]}}
```

### Event `request_completed`

Content-free completion row. It also appears in the bounded status request ring.

```jsonl
{"schema_version":1,"operation_id":"11111111111111111111111111111111","seq":1,"time":"2026-10-13T18:00:00Z","product":{"name":"bubo","version":"0.1.0","revision":"3ff74dce","engine_version":"0.1.0"},"type":"request_completed","payload":{"id":"r1","time":"2026-10-13T18:00:00Z","slot":null,"prompt_tokens":100,"completion_tokens":10,"cached_tokens":0,"ttft_ms":20,"prefill_tps":2000,"decode_tps":50,"cache_kind":"miss","finish_reason":"length","outcome":"completed"}}
```

### Event `support_bundle`

Exact preview text, member hashes and ZIP hash. saved marks whether the new local file was written.

```jsonl
{"schema_version":1,"type":"support_bundle","operation_id":"cca03717e346430eae464aeaae2c2b73","seq":1,"time":"2026-10-14T03:44:45Z","product":{"name":"bubo","version":"0.1.0","revision":"7670503d","engine_version":"0.10.0"},"payload":{"schema_version":1,"saved":false,"zip_bytes":4546,"bundle_sha256":"2baf3bc44c1f85fe1c57d60d8d1f62463542322849425fb67bb872fd42ec2019","files":["versions.json","settings.json","status.json","diagnose.json","history.json","logs.json","manifest.json"],"text_preview":{"versions.json":"{\"schema_version\":1,\"product\":\"bubo\",\"product_version\":\"0.1.0\",\"engine_version\":\"0.10.0\",\"revision\":\"7670503d\",\"binary_sha256\":{\"bootstrap\":\"ee2769e91920baddc0aace4bdac726b9bfb75982ae7d92cd0b842e581d662472\",\"engine\":null}}\n","settings.json":"{\"available\":true,\"version\":1,\"port\":8080,\"context\":262144,\"sessions\":2,\"start_gui_at_login\":false,\"profile\":\"default\",\"last_error_code\":\"device_lost\"}\n","status.json":"{\"available\":true,\"prepared\":false,\"selection_required\":false,\"phase\":\"not_prepared\",\"readiness\":{\"runtime\":false,\"model\":false},\"last_error_code\":\"runtime_missing\"}\n","diagnose.json":"{\"available\":true,\"source\":\"persisted_state\",\"status\":\"WARN\",\"checks\":[{\"name\":\"gpu\",\"status\":\"WARN\"}]}\n","history.json":"[{\"type\":\"diagnose\",\"state\":{\"available\":true,\"status\":\"PASS\",\"checks\":[]}},{\"type\":\"diagnose\",\"state\":{\"available\":true,\"status\":\"PASS\",\"checks\":[]}},{\"type\":\"diagnose\",\"state\":{\"available\":true,\"status\":\"PASS\",\"checks\":[]}},{\"type\":\"diagnose\",\"state\":{\"available\":true,\"status\":\"PASS\",\"checks\":[]}},{\"type\":\"diagnose\",\"state\":{\"available\":true,\"status\":\"PASS\",\"checks\":[]}},{\"type\":\"diagnose\",\"state\":{\"available\":true,\"status\":\"PASS\",\"checks\":[]}},{\"type\":\"diagnose\",\"state\":{\"available\":true,\"status\":\"PASS\",\"checks\":[]}},{\"type\":\"diagnose\",\"state\":{\"available\":true,\"status\":\"PASS\",\"checks\":[]}},{\"type\":\"diagnose\",\"state\":{\"available\":true,\"status\":\"PASS\",\"checks\":[]}},{\"type\":\"diagnose\",\"state\":{\"available\":true,\"status\":\"PASS\",\"checks\":[]}},{\"type\":\"diagnose\",\"state\":{\"available\":true,\"status\":\"PASS\",\"checks\":[]}},{\"type\":\"diagnose\",\"state\":{\"available\":true,\"status\":\"PASS\",\"checks\":[]}},{\"type\":\"diagnose\",\"state\":{\"available\":true,\"status\":\"PASS\",\"checks\":[]}},{\"type\":\"diagnose\",\"state\":{\"available\":true,\"status\":\"PASS\",\"checks\":[]}},{\"type\":\"diagnose\",\"state\":{\"available\":true,\"status\":\"PASS\",\"checks\":[]}},{\"type\":\"diagnose\",\"state\":{\"available\":true,\"status\":\"PASS\",\"checks\":[]}},{\"type\":\"diagnose\",\"state\":{\"available\":true,\"status\":\"PASS\",\"checks\":[]}},{\"type\":\"diagnose\",\"state\":{\"available\":true,\"status\":\"PASS\",\"checks\":[]}},{\"type\":\"diagnose\",\"state\":{\"available\":true,\"status\":\"PASS\",\"checks\":[]}},{\"type\":\"diagnose\",\"state\":{\"available\":true,\"status\":\"PASS\",\"checks\":[]}},{\"type\":\"diagnose\",\"state\":{\"available\":true,\"status\":\"PASS\",\"checks\":[]}},{\"type\":\"diagnose\",\"state\":{\"available\":true,\"status\":\"PASS\",\"checks\":[]}},{\"type\":\"diagnose\",\"state\":{\"available\":true,\"status\":\"PASS\",\"checks\":[]}},{\"type\":\"diagnose\",\"state\":{\"available\":true,\"status\":\"PASS\",\"checks\":[]}},{\"type\":\"diagnose\",\"state\":{\"available\":true,\"status\":\"PASS\",\"checks\":[]}},{\"type\":\"diagnose\",\"state\":{\"available\":true,\"status\":\"PASS\",\"checks\":[]}},{\"type\":\"diagnose\",\"state\":{\"available\":true,\"status\":\"PASS\",\"checks\":[]}},{\"type\":\"diagnose\",\"state\":{\"available\":true,\"status\":\"PASS\",\"checks\":[]}},{\"type\":\"diagnose\",\"state\":{\"available\":true,\"status\":\"PASS\",\"checks\":[]}},{\"type\":\"diagnose\",\"state\":{\"available\":true,\"status\":\"PASS\",\"checks\":[]}},{\"type\":\"diagnose\",\"state\":{\"available\":true,\"status\":\"PASS\",\"checks\":[]}},{\"type\":\"diagnose\",\"state\":{\"available\":true,\"status\":\"PASS\",\"checks\":[]}}]\n","logs.json":"{\"entries\":[{\"event\":\"ready\",\"level\":\"info\",\"count\":2}],\"dropped_lines\":2}\n","manifest.json":"{\"schema_version\":1,\"files\":[{\"name\":\"versions.json\",\"bytes\":223,\"sha256\":\"7ac8137be5bdf384bdaef7ac102fc4d35abda71910aa60d8e5b7942cbf4703c9\"},{\"name\":\"settings.json\",\"bytes\":152,\"sha256\":\"039bb34a9d5b9249458c4a7fdc28d4e649f639e05dc393572369fbcb4d1aea43\"},{\"name\":\"status.json\",\"bytes\":166,\"sha256\":\"ed8d498ac8f53e0136d4dacce2f3a67ee5e44622b915b143be00eeaddecef607\"},{\"name\":\"diagnose.json\",\"bytes\":104,\"sha256\":\"b41c5c172a2d0e1a6a0abdb16eee55226b6c74d8182a035f917735983a0f318f\"},{\"name\":\"history.json\",\"bytes\":2402,\"sha256\":\"ead797f4afaa63e6d86ed67c0def76503c45fbb594b74cd858f8853ce9ebae22\"},{\"name\":\"logs.json\",\"bytes\":75,\"sha256\":\"47515c25e41194f60c4cb7e8bf32eba35ad251499b10b8b3906bde88a4594e9a\"}]}\n"},"sha256":{"versions.json":"7ac8137be5bdf384bdaef7ac102fc4d35abda71910aa60d8e5b7942cbf4703c9","settings.json":"039bb34a9d5b9249458c4a7fdc28d4e649f639e05dc393572369fbcb4d1aea43","status.json":"ed8d498ac8f53e0136d4dacce2f3a67ee5e44622b915b143be00eeaddecef607","diagnose.json":"b41c5c172a2d0e1a6a0abdb16eee55226b6c74d8182a035f917735983a0f318f","history.json":"ead797f4afaa63e6d86ed67c0def76503c45fbb594b74cd858f8853ce9ebae22","logs.json":"47515c25e41194f60c4cb7e8bf32eba35ad251499b10b8b3906bde88a4594e9a","manifest.json":"f54603868a2cf128ad9e90db1faafe6e26703bfcd5054794494ee83c89e7f983"}}}
```

## Additional field names

The registry and nested typed fields use these wire names:

`--allow-unmeasured-memory`, `--api-key`, `--cache-disk`, `--cache-disk-bytes`, `--cache-disk-staging-bytes`, `--cache-ram-bytes`, `--context`, `--debug-hooks`, `--dflash-model`, `--draft-head-model`, `--draft-policy`, `--draft-tokens`, `--dspark-model`, `--exact`, `--frequency-penalty`, `--host`, `--log-level`, `--log-progress`, `--loop-breaker`, `--manifest`, `--max-buffered-output-bytes`, `--max-buffered-output-total`, `--max-connections`, `--max-output-bytes`, `--max-pending`, `--max-pending-per-client`, `--max-request-bytes`, `--max-score-tokens`, `--max-tokens`, `--min-draft-tokens`, `--min-keep`, `--min-p`, `--mmproj`, `--model`, `--mtp-draft-vocab`, `--mtp-model`, `--mtp-policy`, `--no-behaviour`, `--no-cache`, `--no-cache-disk`, `--no-draft-head`, `--no-mtp`, `--no-prompt-lookup`, `--no-twin`, `--no-vision`, `--npu-prefill`, `--prefill-chunk`, `--presence-penalty`, `--preserve-thinking`, `--print-effective-settings`, `--profile`, `--prompt-lookup`, `--reasoning-effort`, `--repeat-last-n`, `--repeat-penalty`, `--request-timeout-ms`, `--root`, `--seed`, `--served-model-name`, `--sessions`, `--speculative`, `--temperature`, `--think`, `--think-marker-bias`, `--top-k`, `--top-p`, `--trace`, `--ttl`, `--twin-model`, `--verbose`, `--verify-model`, `--voice`, `--voice-lang`, `--voice-text`, `A3B_ATTN_CHECK`, `A3B_ATTN_PRECISE`, `A3B_DECODE_ATTN`, `A3B_DFLASH_PROFILE`, `A3B_NO_RESUME`, `A3B_PROBE_ONLY`, `A3B_WIDE_ATTN_CHECK`, `A3B_WIDE_BF16_BLAS`, `A3B_WIDE_BLAS_F32`, `A3B_WIDE_DEBUG`, `A3B_WIDE_F16_MAXCOLS`, `A3B_WIDE_F16_MINROWS`, `A3B_WIDE_FN_ATTN`, `A3B_WIDE_NO_DOWN64`, `A3B_WIDE_NO_F16`, `A3B_WIDE_PROFILE`, `BUBO_API_KEY`, `BUBO_MMPROJ`, `BUBO_MTP_MODEL`, `DS4_LOCK_FILE`, `GPU_MAX_HW_QUEUES`, `GUFO_A3B_DATA`, `GUFO_A3B_MODEL`, `GUFO_BATCH_DRAFT_HEAD`, `GUFO_CACHE_HOST_FLOOR_MB`, `GUFO_CACHE_REMOVAL_LOG`, `GUFO_DEBUG_HOOKS`, `GUFO_DEEPSEEK_DSPARK_TIMING`, `GUFO_DEEPSEEK_DSPARK_TRACE`, `GUFO_DEEPSEEK_DSPARK_WIDTH_TRACE`, `GUFO_DEEPSEEK_ROCM_DECODE_STAGE_PROFILE`, `GUFO_DEEPSEEK_ROCM_GRAPH_PREFILL_PROFILE`, `GUFO_DEEPSEEK_ROCM_GRAPH_TOKEN_PROFILE`, `GUFO_DEEPSEEK_ROCM_INDEXER_STAGE_PROFILE`, `GUFO_DEEPSEEK_ROCM_LAYER_STAGE_PROFILE`, `GUFO_DEEPSEEK_ROCM_Q_STAGE_PROFILE`, `GUFO_DENSE_TWIN_ALLOW_Q8`, `GUFO_DENSE_TWIN_BATCH`, `GUFO_DENSE_TWIN_MAX_N`, `GUFO_DENSE_TWIN_MODEL`, `GUFO_DENSE_TWIN_SHADOW`, `GUFO_DENSE_TWIN_TRACE`, `GUFO_DISK_BOUNDARY_ALIGN`, `GUFO_DISK_DEFER`, `GUFO_DISK_DEFER_MS`, `GUFO_DISK_EVICT_BACKLOG_BYTES`, `GUFO_DISK_EVICT_DOMINATED`, `GUFO_DISK_EVICT_SKIP_TOKENS`, `GUFO_DISK_IDLE_MS`, `GUFO_DISK_INPASS`, `GUFO_DISK_PERSIST`, `GUFO_DISK_SLICE`, `GUFO_DISPATCH_TELEMETRY`, `GUFO_DRAFT_OUT_MODEL`, `GUFO_DRAFT_VOCAB_AUTO`, `GUFO_EOG_BOUNDARY`, `GUFO_EOG_DEFER_P`, `GUFO_EOG_MIN_PROB`, `GUFO_EOG_MIN_PROB_EXACT`, `GUFO_EVAL_API_KEY`, `GUFO_EVAL_ARTIFACT_ID`, `GUFO_EVAL_MODEL_REVISION`, `GUFO_EVAL_SERVER_COMMIT`, `GUFO_EVAL_SERVER_CONFIG`, `GUFO_EVAL_SERVER_LABEL`, `GUFO_EXPERT_V2K`, `GUFO_FFMPEG`, `GUFO_FFPROBE`, `GUFO_GROUPED_DOWN`, `GUFO_HIPBLASLT_PLAN_CACHE`, `GUFO_HOST`, `GUFO_IQ4_WIDE_VARIANT`, `GUFO_IQ4_WMMA_WAVES`, `GUFO_LENIENT`, `GUFO_LENIENT_LAMBDA`, `GUFO_LENIENT_THINK_ONLY`, `GUFO_LOGIT_BIAS`, `GUFO_LOG_SYNC`, `GUFO_LOOP_BREAKER`, `GUFO_LOOP_BREAKER_EOG_CLOSE`, `GUFO_LOOP_BREAKER_TRANSITION`, `GUFO_LOOP_BREAKER_TRANSITION_TEXT`, `GUFO_MAX_IMAGE_TOKENS`, `GUFO_MEM_ACCOUNTING`, `GUFO_MODEL_SUPPRESS_TOKENS`, `GUFO_MOE_BENCH`, `GUFO_MTP_TRACE`, `GUFO_NPU_DIR`, `GUFO_NPU_GATE_SLICES`, `GUFO_NPU_KEEPALIVE`, `GUFO_NPU_KEEPALIVE_MS`, `GUFO_NPU_SHEXP`, `GUFO_NPU_SLICES`, `GUFO_NPU_TIMEOUT_MS`, `GUFO_PLATFORM_TUNING`, `GUFO_PORT`, `GUFO_PREFILL_DIAG`, `GUFO_PREFILL_FORWARD_TOKENS`, `GUFO_PREFILL_STEP_LOG`, `GUFO_PREFILL_TARGET_MS`, `GUFO_PRODUCTION_SAMPLING`, `GUFO_QWEN27_DRAFTLOG`, `GUFO_QWEN27_FLUSH`, `GUFO_QWEN27_LOOKUP`, `GUFO_QWEN27_LOOKUP_MODE`, `GUFO_QWEN27_PROFILE`, `GUFO_REASONING_BUDGET`, `GUFO_REQUIRE_GFX1151`, `GUFO_REQUIRE_GPU`, `GUFO_REQUIRE_HIP`, `GUFO_ROUTED_LANEROW`, `GUFO_SPEC_EOG_STOP`, `GUFO_SURVIVAL_FLOOR`, `GUFO_THINK_EOG_GUARD`, `GUFO_THINK_GUARD_ESCALATE`, `GUFO_THINK_MARKER_BIAS`, `GUFO_THINK_MARKER_BIAS_BY_EFFORT`, `GUFO_THINK_MARKER_LIST`, `GUFO_TRACE`, `GUFO_WEIGHT_UPLOAD_READERS`, `HOME`, `HOST`, `LOCALAPPDATA`, `PATH`, `PORT`, `QWEN3_ASR_ARTIFACTS`, `QWEN3_ASR_MODEL_ROOT`, `QWEN3_ASR_TEST_WAV`, `TEMP`, `XDG_CACHE_HOME`, `aliases`, `also`, `api`, `apiKey`, `audience`, `authenticated`, `available`, `baseUrl`, `bubo`, `built_in`, `bundle_sha256`, `byte_unit`, `bytes`, `cache_kind`, `cached_tokens`, `carve_budget`, `class`, `completion_tokens`, `conditions`, `connection_helpers`, `context`, `contextWindow`, `context_max`, `decode_tps`, `default`, `default_text`, `display_name`, `display_unit`, `echo`, `entropy_gate`, `exact`, `exact_off`, `exclusive_minimum`, `experimental`, `files`, `finish_reason`, `format`, `formula`, `gib`, `group`, `hardware_scope`, `high`, `host_budget`, `host_cache_snapshot_allowance`, `inject_with`, `kind`, `low`, `max`, `max_tokens`, `maximum`, `medium`, `min`, `min_p`, `minimum`, `mmproj`, `model_family`, `model_id`, `models`, `mtp`, `note`, `outcome`, `per_session_state`, `per_slot_context`, `platform`, `port`, `prefill_tps`, `presence_penalty`, `profile`, `profiles`, `providers`, `purpose`, `reasoning_effort`, `reference_coefficient`, `repeat_penalty`, `request_history`, `requires`, `reserve`, `restart_required`, `saved`, `scratch_phase_allowance`, `seed`, `session_state_total`, `settings`, `size_bytes`, `slot`, `source_path`, `state`, `state_history`, `subagents`, `suppressed_by`, `switch_registry`, `table_version`, `takes_value`, `temperature`, `text_preview`, `think`, `tokens_used`, `top_k`, `top_p`, `total_range`, `ttft_ms`, `twin`, `value_rule`, `values`, `weights`, `xhigh`, `zip_bytes`.

## Machine readings

`status --json` and `diagnose --json` add the object `machine`. The capability
`machine_readings` is true when the CLI can report it. All access is read only.
The CLI does not change power state, Variable Graphics Memory (VGM), firmware or
BIOS settings. Old clients ignore the new key.

Each reading has the same fields: `value`, `source`, `api_version`, `unit`,
`sample_time` (UTC), `available` and `reason`. A reading that is not available
has a null `value` and a `reason`. Null never means zero.

`machine` contains `read_only` (always true) and `sample_time`, and these groups.

| Group | Fields |
|---|---|
| `windows` | `overlay_scheme` (GUID text from `PowerGetEffectiveOverlayScheme`), `effective_power_mode`, `power_source` (`ac` or `battery`) |
| `gpu` | `power` (W), `clock` (MHz), `name` |
| `variable_graphics_memory` | `supported`, `carved`, `remaining` (ADLX) |
| `dxgi` | `adapter_name`, `dedicated_video_memory`, `shared_system_memory`, `budget`, `current_usage`, `available_for_reservation`, `agree` (Windows DXGI) |
| `host_physical_memory` | Total RAM in bytes |
| `oem` | `manufacturer`, `model`, `bios_vendor`, `bios_version`, `bios_date` |
| `recipe_suggestion` | `text`, `tested_profile`, `reason`, `inputs` |

`effective_power_mode` has `raw_value`: the integer that Windows reports through
`PowerRegisterForEffectivePowerModeNotifications` (version 2). It is the same
source as the `effective_power_mode` field of the adlx-gpu-monitor harness.
`windows_enum_name` is the name of that integer in the Windows API enumeration,
and `windows_enum_source` names the header. This name is not an ASUS or
Armoury Crate mode name. `validated_mapping` is always null, and
`validated_mapping_reason` says why: no measured mapping exists. The
`reference_observation` object records one observation from the harness. On
the ASUS ProArt PX13 measurements showed raw 4 in Armoury Crate Performance and raw
1 in Whisper. `observed_meaning` holds the first, and `other_observed` holds
the second. Its `confidence` is "harness-observed, ASUS PX13 only" and
`validated` is false. `applies_to_this_machine` is true only for an ASUS ProArt
PX13 model. `matches_reference` is then true when `raw_value` is 4, and false
for other values. It is null when the observation does not apply or the reading
is not available. Show the warning "Power mode is below Performance" only when
`applies_to_this_machine` is true and `matches_reference` is false. Do not
compare enumeration numbers as a ranking.

`gpu` readings come from the AMD driver (ADLX), and only when the support flag
in `support_flag` is true (`supported` repeats it). `label` is "GPU power" or
"GPU clock", and `metric` is the ADLX metric name. `validation` says the power
value is not validated against a meter. It can include more than the graphics
block on an APU. Do not show it as package power.

`carved` and `remaining` are in bytes. ADLX reports them as a double in "GB" and
does not name the base. The CLI uses `raw_value` x 2^30, shows the original
number as `raw_value`, names its unit in `raw_unit`, and gives `gib` and
`conversion`. On the test laptop the carve plus the remaining RAM equals the
physical RAM, which fits the 2^30 base. The carve is a reservation, not free GPU
memory.

`dxgi` reads the same memory through the Windows graphics layer. It needs no
AMD driver library and no SDK, so a shipped build can report it. The CLI lists
the DXGI adapters and uses the AMD hardware adapter that has the largest
`DedicatedVideoMemory`. It never uses a software adapter (WARP) and never uses
a fixed adapter index. Each field has the reading shape above, plus `metric`
(the Windows field name) and `gib`. `dedicated_video_memory` comes from
"DXGI IDXGIAdapter::GetDesc1 DedicatedVideoMemory" and
`shared_system_memory` from "DXGI IDXGIAdapter::GetDesc1 SharedSystemMemory".
`budget`, `current_usage` and `available_for_reservation` come from
"IDXGIAdapter3::QueryVideoMemoryInfo" (local segment group). If the adapter
has no `IDXGIAdapter3`, or the call fails, these three are unavailable with a
reason. A dedicated size of 0 is unavailable, never a zero carve.

`dxgi.agree` is true when `dxgi.dedicated_video_memory` and
`variable_graphics_memory.carved` differ by 1 MiB (1048576 bytes) or less. It is
false when they differ by more. It is null, with `agree_reason`, when one of
the two readings is not available. `agree_compares` names the two readings, and
`agree_tolerance_bytes` is the 1 MiB limit. A false value means the two readers do not
describe the same carve. Do not hide it. The `dxgi` object is additive and
optional for old readers. An old fixture without `dxgi` stays valid.

`recipe_suggestion.text` is "No tested profile is available for this memory
setting." until a tested recipe record exists. `tested_profile` is null
and `inputs` carries `carved_bytes` and `host_physical_bytes`.

ADLX loads at run time from the installed AMD driver. If it is missing, the
readings are unavailable with a reason. A build that has no ADLX support
reports that reason. The shipped build has no ADLX support because the AMD
licence does not allow the SDK headers in the package. The `dxgi` readings do
not depend on ADLX. Status and prepare do not depend on ADLX. The AMD SDK
and driver files are not part of the package.

The command `bubo power --overlay` is not built. No code path sets the Windows
power overlay. It stays unbuilt until the measured gate in the plan passes.

## Fixture directory index

`tests/cli/contract/fixture-index.json` indexes valid, invalid, input-valid,
input-invalid and v1-legacy fixtures. v1-legacy retains J1a/J1b/J2a valid
examples byte-for-byte. Run compatibility_test.ps1 to prove old validation,
unknown-key acceptance, missing-required rejection and index coverage.
The generated switches.golden.json, switches.schema.json, engine-panels.json
and pi-snippet.schema.json supplement that index.
