# Install Halowin Bubo on Windows

**Supported hardware:** Bubo is built and tested for AMD Strix Halo with 128 GB of memory on Windows, mainly laptops. It can work on other machines, but we make no promise.

Use this setup for a Strix Halo laptop with Radeon 8060S graphics at approximately 70 W.
Bubo runs natively on Windows 11 x64 with AMD HIP.

## Prepare the laptop

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

1. Download `bubo-v0.1.0-windows-x64-gfx1151.zip` from the [Bubo release page](https://github.com/haloforge-labs/bubo/releases/tag/v0.1.0). The file is 44,096,931 bytes. Its SHA-256 value is `d1c92dacf68e8e379b8e02c8d68ba18ba102b2fd22784e25cb06b0757fac1306`. After you extract it, `bubo --version` prints `bubo 0.1.0 (3ff74dce), engine gufo 0.10.0`.
2. Extract the complete package into a folder.
3. Open PowerShell in that folder.

The release command is `bubo`.
PowerShell uses `.\bubo.exe` when the executable is in the current folder.

## Windows security prompts

Like many open-source Windows builds, Bubo v0.1.0 is not code-signed. Windows can show warnings.

- A zip that a browser downloaded carries a "downloaded from the internet" mark. Right-click the zip, select **Properties**, tick **Unblock**, select **OK**, then extract it.
- If SmartScreen shows "Windows protected your PC" for `start.cmd`, select **More info**, then **Run anyway**.
- If Smart App Control is on (common on new Windows 11 installs), Windows blocks unsigned apps and offers no per-app override. Bubo v0.1.0 does not run while Smart App Control is on.

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

The total download is 1,014,229,359 bytes (967.2 MiB).
Before the download, the command shows the address, exact size and full SHA-256 value of each wheel, the destination folders and the AMD licence information. It then asks for your approval. If you decline, nothing is installed (exit code 3).
The `--yes` option accepts the displayed plan. It prints the same text first.
The command checks the SHA-256 value of each archive before it extracts any file.
It installs 13 DLLs and the `gfx1151` data files (about 303 MiB) beside `bubo.exe`. It saves the licence texts in the `licenses` folder.
The step needs no Python and no administrator access.

By default the runtime goes beside `bubo.exe`. Use `--runtime-dir PATH` to choose another folder. The downloaded wheels stay in a cache folder (default `ROOT\cache\rocm-10.0.0-gfx1151`; change it with `--cache-dir PATH`).

If a download stops, run the same command again. The command resumes partial downloads that it started. Use `--no-resume` to start them again.

If your network needs a proxy, set `HTTPS_PROXY` before you run the command. Set `NO_PROXY` for addresses that must not use the proxy.

```powershell
$env:HTTPS_PROXY = "http://proxy.example:8080"
```

### AMD licence terms

Bubo does not include AMD files. The runtime step of `bubo prepare` downloads AMD's ROCm 10.0.0 packages for `gfx1151` directly from `stable.repo.amd.com`. Before it downloads, it shows the address, size and SHA-256 value of each package, and the licence information. AMD licenses ROCm per component, and components can have third-party terms. Read the [AMD ROCm 10.0.0 licence page](https://rocm.docs.amd.com/en/docs-10.0.0/about/license.html) and the licence texts that Bubo saves in the `licenses` folder. Other AMD terms can apply to these files. If you continue, you accept the terms that apply.

The pinned runtime is ROCm 10.0.0 for `gfx1151`.
ROCm 10.1.0 does not match the tested runtime files.

## Get the model

Model files have separate licences.
Read the [Qwen licence](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE) before the download. The command also saves the licence and notice files with the model.

```powershell
.\bubo.exe prepare model
```

The command downloads eight pinned files and checks each one against its SHA-256 value. It builds nothing on your machine.

| Files | Source | Size |
| --- | --- | ---: |
| Four `UD-Q4_K_XL` shards | [Huihui GGUF repository](https://huggingface.co/huihui-ai/Huihui-Qwen3.8-Flash-Next-abliterated-GGUF/tree/73e9eb7c69fdbf667def63e233e70845e7d5ac0b/UD-Q4_K_XL) (revision `73e9eb7c`) | 111,334,655,392 bytes (103.7 GiB) |
| MTP predictor and vision projector | [Unsloth base repository](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF) | 3,694,111,200 bytes (3.4 GiB) |
| Dense twin and Q4_K draft head | HaloForge Labs Hugging Face repository (commit `bb72204d`) | 1,588,370,016 bytes (1.5 GiB) |

The total is 116,617,136,608 bytes (108.6 GiB). Before it downloads, the command shows this plan and asks for approval. Use `--yes` to accept the plan.

The twin and the draft head are derived from the Huihui `UD-Q4_K_XL` shards. They match only that source model. Both use the Qwen Community License 1.0, not MIT. The source model is an abliterated model; read the usage warnings on its model card.

By default the files go in `%LOCALAPPDATA%\Bubo`. Use `--output ROOT` to choose another folder, for example `.\bubo.exe prepare model --output "D:\BuboData"`. The model folder is `ROOT\models\huihui-qwen38-flash-next-ud-q4-k-xl-v1\`. The command writes `bubo-package.json` last. The server finds the other files through that file.

Plan about 113 GiB of free space for the runtime and the model together (an estimate).

To prepare the runtime and the model with one approval, run `.\bubo.exe prepare` with no step name.

If a download stops, run the same command again. The command resumes. Use `--no-resume` to start again.

### Use files you already have

1. Put all four shards in the model folder with the exact names from the command output.
2. Run `.\bubo.exe prepare model --output ROOT`. The command checks the files and downloads only what is missing.
3. Use `--mtp-model PATH` or `--mmproj PATH` to copy an existing MTP or vision file. The file must match the pinned hash.
4. Use `--offline` to use local files only. A missing file then fails before any network use (exit code 4).

An existing file that does not match is never overwritten. The error names the file. Delete the file and run the command again.

The base Qwen `UD-Q4_K_XL` model also works with explicit file paths. Bubo then runs without the twin and the draft head. Use one source model at a time.

## Model examination

The default checks find a wrong file, a wrong model variant and a truncated file. They use file sizes, the GGUF header and tensor table, and a quick fingerprint of the tensors.
The server never reads every model byte at startup, including the first start.

To check every byte, add `--verify` to the prepare command. It forces a full SHA-256 check of the selected files. The server option `--verify-model` does the same for the eight prepared files before the server starts. On an SSD this takes about 2 to 4 minutes (an estimate).


## Notices

1. Read `NOTICE-windows.md` in the package.
2. Read `THIRD_PARTY_NOTICES.md`.
3. Keep the `licenses` folder beside the installed runtime.

Gufo and the pixmaate Windows layer retain the MIT licence.
AMD components and model files retain their separate licences.

Continue with the [quickstart](quickstart.md).
