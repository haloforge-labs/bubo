# Operator commands

Run from the job directory in PowerShell 7. The implementation job does not start a server or use the GPU.

The default gate loads five arms in order: `huihui`, `ista-off`, `ista-sites-fit`, `ista-sites-1.0`, and `ista-stream-1.0`. Each receives the same 50 AdvBench prompts, greedy, thinking off, with at most 160 generated tokens. The stream arm uses layers 4–44. Allow about **45–55 minutes**, plus lock wait: roughly 7–11 minutes per arm, with the Huihui load potentially longer. Use `-SkipStream` to omit the final arm and run the four-arm comparison.

```powershell
$sha = (git -C <repo> rev-parse --short=8 HEAD).Trim()
$Runtime = "<runtimes>\bubo-lqua26-$sha"
$Paths = @{
  Direction   = '<directions>\refusal-r-d180ec15-fitted.gguf'
  Ista        = '<ista-iq3s>\Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf'
  Out         = '<results>\operator-<timestamp>'      # new directory
  PackageDir  = '<huihui-package>'                    # holds mtp\ and draft_head\
  HuihuiModel = '<huihui-shards>\Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf'
  GpuLock     = '<run>\gpu-harness.lock'
  TempDir     = '<scratch>\tmp'
  AdvBenchCsv = '<advbench>\harmful_behaviors.csv'
}
.\operator-lqua26.ps1 -Runtime $Runtime @Paths
# Optional: omit the last arm
.\operator-lqua26.ps1 -Runtime $Runtime @Paths -SkipStream
# Optional: pass -BuildLockScript <wrapper.ps1> to run inside a shared build/speed lock
```

Every `<...>` is a path on the machine that runs the gate.

The runner refuses an existing server, obtains the shared build/speed lock, waits for and exclusively creates the GPU-harness lock, and checks again for servers after waiting. It clears and restores GUFO/BUBO variables, stops only its matching child, and aborts on missing or wrong projection counters. The Huihui arm uses production shard hard links and explicit production MTP and draft-head paths. The ISTA arms use the native IQ3_S shards.

The direction is the fitted file `refusal-r-d180ec15-fitted.gguf`, passed with `-Direction <path>`. It contains the same unit vector as the first-pass file and a fitted strength of **1.955837064**, the median of the 96 per-site through-origin regressions. A version-1 file without strength metadata uses that same fitted value for compatibility. A malformed file or incomplete site list stops loading.

| Setting | Values and default |
| --- | --- |
| `--refusal-direction` | `off` by default; `on` enables the selected file. |
| `--refusal-direction-file` | Path to the direction file. Supplying a path alone leaves the feature off. |
| `--refusal-direction-strength` | Finite number from 0 to 4; omitted uses the fitted value in the file. `1.0` removes the direction and `0` has no effect. |
| `--refusal-direction-mode` | `sites` by default, or `stream`. |
| `--refusal-direction-layer-start`, `--refusal-direction-layer-end` | Inclusive stream range, default 4–44. Valid range is 1–47; ignored in sites mode. |

The legacy `GUFO_REFUSAL_DIRECTION=<file>` still enables the feature when no explicit on/off setting is supplied. The debug environment overrides for strength, mode, and the two layer bounds use the `GUFO_REFUSAL_DIRECTION_` prefix; matching CLI options take precedence. `--refusal-direction off` takes precedence over a legacy file setting. Settings are fixed for the lifetime of the loaded model and require a restart.

Inspect complete classification-flip answers printed before counts, also saved in `classification-flips.txt`. All full answers, token IDs, usage and HTTP responses remain in the arm JSONL files. `readout.txt` gives refusal rates and, against Huihui, first-token agreement and the common-prefix length of the first 40 generated tokens. Review the raw flips before interpreting the counts. Agreement and refusal rates are measurements, not predetermined pass thresholds.

Optional `-Reasoning` adds four short reasoning prompts with two seeds to every ISTA arm. Optional `-Speed` adds a warm-up and three timing requests per ISTA arm. Allow about **15–40 minutes** and **8–15 minutes**, respectively, depending on generation lengths. They reuse the gate's model loads.

Counters describe host submissions, including graph captures; graph replays do not increment them. Sites mode fuses projection into existing writer consumers, with no added projection launches. Stream mode adds one projection launch per selected layer per forward and recomputes the following normalization and inject partials. Its actual speed cost is unmeasured. The MTP draft block is not projected.

CPU checks and the reproducible metadata fit:

```powershell
.\cpu.ps1 -Label controls -Regex 'qwen38_flash_next\.(config|decode_cycle|mtp_sampling|refusal_direction)$|^(switches_registry_test|switches_recipe_test)$'
.\operator-policy-test.ps1
python -B <repo>\tests\models\qwen38_flash_next\refusal_real_test.py --direction <directions>\refusal-r-d180ec15-fitted.gguf
python -B <repo>\tools\derive_refusal_r.py --fit-existing --out <directions> --direction-file <directions>\refusal-r-d180ec15.gguf
```

The focused CTest selection excludes the known brand-scan failure in the full `release_profile_test`; its registry and settings checks were also run separately. The last command reads the saved `derivation.json` and original direction vector, writes the fitted copy and `derivation-fitted.json`, and does not reread model weights. The real-tensor CPU comparison takes about one minute. To repeat the full derivation, omit `--fit-existing` and pass `--huihui-dir`, `--ista`, `--cross-check` and `--scratch`; it uses the scratch directory you name (put it on a drive with room) and the 6 GiB free-memory guard.

Rebuild only through the shared lock at `-j4`:

```powershell
.\build.ps1 -Phase Configure
.\build.ps1 -Phase Build
.\build.ps1 -Phase Bootstrap
```
