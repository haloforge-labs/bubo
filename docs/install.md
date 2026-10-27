# Install Halowin Bubo on Windows

**Supported hardware:** Bubo is built and tested for AMD Strix Halo with 128 GB of memory on Windows, mainly laptops. It can work on other machines, with no promise.

**96 GB machines:** use the lower-memory model. Set Variable Graphics Memory (AMD Software: Adrenalin Edition > Performance > Tuning) to the largest value offered (72 GB if available, at least 64 GB), then restart Windows. At 72 GB, Bubo uses the default 200,000-token context for one session. At 64 GB it does too when about 16 GB of system memory is free, and picks a smaller context when less is free. These values are calculated: Bubo has not yet run on a 96 GB machine.

Use this setup for a Strix Halo machine. The speeds on this site come from a laptop with Radeon 8060S graphics at approximately 70 W.
Bubo runs natively on Windows 11 x64 with AMD HIP.

## Prepare the machine

The documented Qwen3.8 Flash Next setup uses 128 GB of system RAM.
Deep context and images need more memory than short text requests.

1. Set GPU memory to 96 GB in the BIOS or the laptop control app.
2. Set the page file initial size to 131072 MB on the system drive.
3. Set the page file maximum size to 131072 MB.
4. Select Performance mode.
5. Install AMD graphics driver 32.0.31041.1004 (2026-08-16) or newer.
6. Install the [Microsoft Visual C++ 2015-2022 Redistributable (x64)](https://aka.ms/vs/17/release/vc_redist.x64.exe). Most PCs already have it. Install it before you run `bubo.exe`: Windows stops `bubo.exe` at start if the runtime is missing. The start script checks for it. The Bubo package does not contain the Microsoft runtime files.

Windows 11 build 26200 is the tested operating system.
The GPU memory setting can have the name “UMA frame buffer” or “carve-out”.
A page file does not replace physical RAM for full-speed operation.

## Get the package

1. Download `bubo-v0.1.1-windows-x64-gfx1151.zip` from the [Bubo release page](https://github.com/haloforge-labs/bubo/releases/tag/v0.1.1). The file is 66,241,059 bytes. Its SHA-256 value is `eeffccd3dc4c322b968a332ce378ffd99521eb377db262ac195111d847d5f2ee`. After you extract it, `bubo --version` prints `bubo 0.1.1 (0cb4bd87), engine gufo 0.10.0`.
2. Extract the complete package into a folder.
3. Open PowerShell in the `bin` folder of the package. The `bubo.exe` file is in `bin`. The file `start.cmd` is in the package folder, one level up.

The release command is `bubo`.
PowerShell uses `.\bubo.exe` when the executable is in the current folder.

## Prepare the AMD runtime

The Bubo package carries no AMD files. The `bubo prepare` command downloads them.

```powershell
.\bubo.exe prepare runtime
```

The command downloads these official AMD wheels from [AMD's ROCm index](https://stable.repo.amd.com/rocm/whl-next/).

| Component | Pinned version |
| --- | --- |
| `rocm_sdk_core` | `10.0.0` |
| `rocm_sdk_libraries` | `10.0.0` |
| `rocm_sdk_device_gfx1151` | `10.0.0` |

The total download is 1,014,229,359 bytes (1.0 GB).
Before the download, the command shows the address, exact size and full SHA-256 value of each wheel, the destination folders and the AMD license information. It then asks for your approval. If you decline, nothing is installed (exit code 3).
The `--yes` option accepts the displayed plan. It prints the same text first.
The command checks the SHA-256 value of each archive before it extracts any file.
It installs 13 DLLs and the `gfx1151` data files (about 318 MB) in the runtime folder. It saves the license texts in the `licenses` folder inside the runtime folder.
The step needs no Python and no administrator access.

By default the runtime goes in the shared folder `%LOCALAPPDATA%\Bubo\runtime\rocm-10.0.0-gfx1151`. A new package folder therefore keeps the runtime. Use `--runtime-dir PATH` to choose another folder, and set `BUBO_RUNTIME_DIR` to the same path when you start the server. A runtime beside `bubo.exe` (the v0.1.0 layout) still works. The downloaded wheels stay in a cache folder (default `ROOT\cache\rocm-10.0.0-gfx1151`; change it with `--cache-dir PATH`).

If a download stops, run the same command again. The command resumes partial downloads that it started. Use `--no-resume` to start them again.

If your network needs a proxy, set `HTTPS_PROXY` before you run the command. Set `NO_PROXY` for addresses that must not use the proxy.

```powershell
$env:HTTPS_PROXY = "http://proxy.example:8080"
```

### AMD license terms

Bubo does not include AMD files. The runtime step of `bubo prepare` downloads AMD's ROCm 10.0.0 packages for `gfx1151` directly from `stable.repo.amd.com`. Before it downloads, it shows the address, size and SHA-256 value of each package, and the license information. AMD licenses ROCm per component, and components can have third-party terms. Read the [AMD ROCm 10.0.0 license page](https://rocm.docs.amd.com/en/docs-10.0.0/about/license.html) and the license texts that Bubo saves in the `licenses` folder. Other AMD terms can apply to these files. If you continue, you accept the terms that apply.

The pinned runtime is ROCm 10.0.0 for `gfx1151`.
ROCm 10.1.0 does not match the tested runtime files.

## Get the model

Model files have separate licenses.
Read the [Qwen license](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE) before the download. The command also saves the license and notice files with the model.

```powershell
.\bubo.exe prepare model
```

The command downloads eight pinned files and checks each one against its SHA-256 value. It builds nothing on your machine.

| Files | Source | Size |
| --- | --- | ---: |
| Four `UD-Q4_K_XL` shards | [Huihui GGUF repository](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF/tree/73e9eb7c69fdbf667def63e233e70845e7d5ac0b/UD-Q4_K_XL) (revision `73e9eb7c`) | 111,334,655,392 bytes (111 GB) |
| MTP predictor and vision projector | [Unsloth base repository](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) | 3,694,111,200 bytes (3.7 GB) |
| Dense twin and Q4_K draft head | HaloForge Labs Hugging Face repository (commit `bb72204d`) | 1,588,370,016 bytes (1.6 GB) |

The total is 116,617,136,608 bytes (117 GB). Before it downloads, the command shows this plan and asks for approval. Use `--yes` to accept the plan.

The twin and the draft head are derived from the Huihui `UD-Q4_K_XL` shards. They match only that source model. Both use the Qwen Community License 1.0, not MIT. The source model is an abliterated model; read the usage warnings on its model card.

By default the files go in `%LOCALAPPDATA%\Bubo`. Use `--output ROOT` to choose another folder, for example `.\bubo.exe prepare model --output "D:\BuboData"`. The model folder is `ROOT\models\huihui-qwen38-flash-next-ud-q4-k-xl-v1\`. The command writes `bubo-package.json` last. The server finds the other files through that file.

Plan about 121 GB of free space for the runtime and the model together (an estimate).

To prepare the runtime and the model with one approval, run `.\bubo.exe prepare` with no step name.

If a download stops, run the same command again. The command resumes. Use `--no-resume` to start again.

### Use files you already have

1. Put all four shards in the model folder with the exact names from the command output.
2. Run `.\bubo.exe prepare model --output ROOT`. The command checks the files and downloads only what is missing.
3. Use `--mtp-model PATH` or `--mmproj PATH` to copy an existing MTP or vision file. The file must match the pinned hash.
4. Use `--offline` to use local files only. A missing file then fails before any network use (exit code 4).

If a file in the model folder does not match its pinned size or SHA-256 value, the plan lists it as `replaces a file that did not match` and adds its size to the download. Keep free space for the whole new file: the old copy stays until the new file passes the check. The command asks the usual question, "Download and prepare all files in this plan? [y/N]", and `--yes` accepts it. Bubo downloads the new file and then replaces the damaged one. If you answer no, Bubo changes nothing (exit code 3).

Bubo does not replace a file that you supplied (for example with `--mmproj`), or a file whose real location is outside the model folder (a link or junction). The error names the file. Delete the file and run the command again. If a running Bubo server holds a file open, the command stops with exit code 7. Stop the server and run the command again.

The base Qwen `UD-Q4_K_XL` model also works with explicit file paths. Bubo then runs without the twin and the draft head. Use one source model at a time.

## Model examination

The default checks find a wrong file, a wrong model variant and a truncated file. They use file sizes, the GGUF header and tensor table, and a quick fingerprint of the tensors.
The server never reads every model byte at startup, including the first start.

To check every byte, add `--verify` to the prepare command. It forces a full SHA-256 check of the selected files. The server option `--verify-model` does the same for the eight prepared files before the server starts. On an SSD this takes about 2 to 4 minutes (an estimate).

## Use the original (non-abliterated) model

The default model is an abliterated build. If you need the original Qwen3.8 Flash Next, prepare the original set. It uses the plain Unsloth shards with the base twin.

1. Make sure that you have about 119 GB of free disk space. The download is 116,617,136,000 bytes (108.608 GiB).
2. Run the prepare command.

```powershell
.\bubo.exe prepare model --model-set original
```

Add `--output ROOT` to choose the parent folder, as for the default model. The command shows the plan first: the files, their sizes, the license texts and the free space. Nothing downloads before you approve.

The set has eight files: the four `UD-Q4_K_XL` shards of [unsloth/Qwen3.8-Flash-Next-GGUF](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) (revision `766911a6`), the MTP predictor, the vision projector, and the base twin and the draft head from [HaloForgeLabs/bubo-qwen3.8-flash-next](https://huggingface.co/HaloForgeLabs/bubo-qwen3.8-flash-next) (revision `0a3c343b`). The model license is the Qwen Community License 1.0. The files go to `ROOT\models\qwen38-flash-next-base-ud-q4-k-xl-v1\`. The default `ROOT` is `%LOCALAPPDATA%\Bubo`.

3. Start the server with the manifest of the original set.

```powershell
.\bubo.exe serve llm --manifest "$env:LOCALAPPDATA\Bubo\models\qwen38-flash-next-base-ud-q4-k-xl-v1\bubo-package.json" --served-model-name bubo
```

Preparing the original set does not change the saved default model. The default stays the model that you prepared first. You must pass `--manifest` each time you want the original set. `bubo status` shows the original set as not prepared, because status checks only the saved default.

The default profile applies on its own. To check that the server bound the twin and the draft head, add `--print-effective-settings` to the command. It prints `twin_bound` and `draft_head_bound`.

The base twin and the draft head work only with these shards. The Huihui twin does not fit them. The draft head is the same file for both models. On the original model, the twin and the draft head raised generation speed by 12.4 % (46.06 to 51.76 tok/s) in a test on an earlier build. See the [model card](https://huggingface.co/HaloForgeLabs/bubo-qwen3.8-flash-next) for the quality checks.

## Notices

1. Read `NOTICE-windows.md` in the package.
2. Read `THIRD_PARTY_NOTICES.md`.
3. Keep the `licenses` folder in the runtime folder.

Gufo and the pixmaate Windows layer retain the MIT license.
AMD components and model files retain their separate licenses.

Continue with the [quickstart](quickstart.md).
