# Choose a Bubo profile

Use `default` for normal use on a 70 W Strix Halo laptop.
Bubo selects it automatically.

Bubo is tuned for Qwen3.8 Flash Next. The other models run with upstream gufo settings.
The full exact controls apply to Qwen3.8 Flash Next only.
For other model families, `exact` supplies the seed rule and keeps their existing settings.

## Profile settings

| Item | `default` | `stock` | `exact` | `subagents` |
| --- | --- | --- | --- | --- |
| Purpose | Normal use | Plain gufo | Reproducible-output mode | Concurrent agent requests |
| Sessions | 1 | Upstream default | 1 | 4 |
| Context per session | 200000 | Model native | 200000 | 131072 |
| RAM cache budget | 8.6 GB | Upstream automatic budget | Disabled | 4.3 GB |
| Disk prompt cache | Off | Off unless set | Disabled | Off |
| Default effort | Medium | Upstream default | Medium | Medium |
| Thinking | On | Model default | On | On |
| Speculative backend | MTP with predictor | Upstream settings | Off | MTP with predictor |
| MTP policy | Survival | Upstream settings | Unused | Survival |
| MTP draft vocabulary | Auto | Upstream settings | Unused | Latin |
| Prompt lookup | On with MTP | Upstream settings | Off | On with MTP |
| Behavior policies | On | No profile defaults | Off | On |
| Lenient acceptance | Bounded with MTP | No profile defaults | Off | Bounded with MTP |
| Dense twin | With matching file | No profile defaults | Disabled | With matching file and batched twin enabled |
| Q4_K draft head | With matching file and MTP | No profile defaults | Disabled | With matching file and MTP |
| Seed | Random | Upstream default | 1 unless supplied | Random |

After `bubo prepare`, the manifest supplies the sidecar files. A flag or an environment value overrides the manifest.
The `stock` profile ignores the manifest twin, draft head and MTP unless you select them. Use `--no-twin`, `--no-draft-head`, `--no-vision` or `--no-mtp` to leave out one file.

## Default

Use this profile for normal chat, code tasks, and agent work.
Its optimal setup includes the MTP predictor, dense twin, and Q4_K draft head.
Medium effort selects marker bias `-1` in the map `low:-2,medium:-1,xhigh:0`.
Bounded lenient acceptance applies inside thinking spans.
The behavior set includes the engine loop breaker.

The default output is not byte-exact.
Without an MTP predictor, the source skips MTP defaults and uses plain autoregressive decode.
Without a manifest or explicit paths, the twin and the draft head do not load.

## Stock

Select this profile with `--profile stock`.
It injects no Bubo environment values or flag defaults.
Existing environment values and explicit flags still apply.
Plain gufo behavior needs an environment without Bubo tuning overrides.

## Exact

Select this profile with `--profile exact`.
The alias `--exact` selects the same profile.

The exact profile pins the settings that make output repeatable: one session, a seed, a fixed prefill size, no speculation and no caches.
Keep the runtime files, model files, machine, prompt bytes, and request settings the same.
The guarantee does not extend across builds, hardware, or other engines.

Exact disables these features.

- Speculative decode and prompt lookup.
- Behavior policies and lenient acceptance.
- Dense twin and draft head.
- RAM and disk prompt caches.

It pins one session and a prefill forward size of 4096 tokens.
The seed is 1 unless you supply another nonnegative seed.
It refuses settings that enable a disabled feature or change the fixed geometry.
If you set a flag or an environment value that would change an exact-profile value, the server stops with a one-line error, for example `Error: profile=exact: explicit value re-enables GUFO_DENSE_TWIN_MAX_N`. The name in the message changes with the setting. Remove that setting, or do not combine it with `--profile exact`.
Each request processes its full prompt, so long multi-turn sessions are slower in exact mode.
The exact profile runs with the prompt cache off (RAM and disk). In the measured exact run every request reported zero cached tokens. The cache RAM budget setting has no effect in this profile.
Cached restores are outside the exactness guarantee. In tests (temperature 1, seed 1) they were not bit-identical to a fresh prefill, including a restore at an unaligned point (4090 tokens). That is why exact turns the cache off.
Use `default` for routine agent work.

### What was tested

In exact mode, Bubo gives the same tokens as the stock reference. This was tested on the release source tree. This is a result for one source tree. It is not a promise for all builds.

1. `--exact` was tested against the stock-flag reference on 24 fixed cases: short, 8K and 33K natural, thinking, tool, stop, 4095, 4096 and 4097 token edges, image, long decode and non-ASCII. The tests used temperature 0 and 1 and seed 1. There were five exact passes and one retained pass, and in each pass all 24 cases matched the reference. The comparison covered the text, the reasoning, the tool calls, the token ids and the finish reasons.
2. Logprobs were compared only where the product supports them (thinking-off cases). The thinking, tool, stop and image cases were compared by text and token ids.
3. The prompt and continuation cache is off in `--exact` (RAM and disk), and `cached_tokens` was 0 on every row. Cached restores are not bit-identical to a fresh prefill (3 of 3 measured cases differed) and are outside this guarantee.
4. The full exact proof ran on build `163762d9`. The release revision `3ff74dce` differs from that tree only in the Windows launcher script and its test. The test used one machine: Windows, AMD Radeon 8060S (Strix Halo), ROCm path. Other hardware, seeds and prompts are not covered. No speed claim.
5. The `exact` and `stock` profiles refuse the NPU and grouped expert kernel switches, and the profile rejects the production environment overrides that were tested (`GUFO_DENSE_TWIN_MAX_N`). Other user overrides were not tested.

## Subagents

Select this profile with `--profile subagents`.
It allocates four sessions with 131072 context tokens each.
It enables the batched twin control and uses a 4.3 GB RAM cache.
The total measured context limit is 524288 tokens.

Four deep concurrent sessions can reach the host RAM guard near 50K tokens per session.
Keep the profile limits unless your task needs less context.
Reduce context when host RAM is limited.
`--allow-unmeasured-memory` permits a total context above the measured limit.
Keep that option off for normal use.

## Overrides

Explicit flags and real environment values override profile defaults.
Exact refuses incompatible overrides.
Environment variable names keep the `GUFO_*` prefix.
Empty environment values count as unset and produce a warning.

1. Print the resolved settings before you change a profile.

```powershell
.\bubo.exe serve llm --model "D:\Models\first-shard.gguf" --profile default --print-effective-settings
```

See the [quickstart](quickstart.md) and [options](options.md).
